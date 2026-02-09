"""
Cerberus SQLite Database Manager
Autonomous edge storage for all rover data: sensor readings, detections,
mission logs, health snapshots. WAL mode for concurrent read/write.
Thread-safe with connection-per-thread pattern.
"""

import os
import sqlite3
import threading
import logging
from typing import Any, Optional
from contextlib import contextmanager

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class CerberusDB:
    """
    Thread-safe SQLite manager for Cerberus.
    Each thread gets its own connection. WAL mode enables
    concurrent reads while one thread writes.
    """

    _instance: Optional["CerberusDB"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[CerberusConfig] = None) -> "CerberusDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if CerberusDB._initialized:
            return

        if config is None:
            config = CerberusConfig()

        self._db_path: str = config.require("database", "path")
        self._journal_mode: str = config.get("database", "journal_mode", default="WAL")
        self._busy_timeout: int = config.get("database", "busy_timeout_ms", default=5000)
        self._local: threading.local = threading.local()
        self._write_lock: threading.Lock = threading.Lock()

        self._ensure_directory()
        self._init_schema()

        CerberusDB._initialized = True
        logger.info("Database initialized at %s (journal=%s)", self._db_path, self._journal_mode)

    def _ensure_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        db_dir: str = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self._db_path, timeout=self._busy_timeout / 1000)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = ?", (self._busy_timeout,))
                conn.execute(f"PRAGMA journal_mode = {self._journal_mode}")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA synchronous = NORMAL")
                self._local.connection = conn
            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to connect to database: {e}") from e
        return conn

    @contextmanager
    def _transaction(self):
        """Context manager for write operations with automatic commit/rollback."""
        conn: sqlite3.Connection = self._get_connection()
        with self._write_lock:
            try:
                yield conn
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise DatabaseError(f"Transaction failed: {e}") from e
            except Exception as e:
                conn.rollback()
                raise

    def _init_schema(self) -> None:
        """Create all tables if they don't exist. Idempotent."""
        schema: str = """
            CREATE TABLE IF NOT EXISTS health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                cpu_temp_c REAL,
                cpu_usage_pct REAL,
                memory_usage_pct REAL,
                disk_usage_pct REAL,
                battery_voltage REAL,
                battery_current_a REAL,
                battery_pct REAL,
                pi_rail_voltage REAL,
                motor_current_a REAL,
                gps_lat REAL,
                gps_lon REAL,
                gps_fix INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                sensor_type TEXT NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                gas_resistance_ohms REAL,
                co2_ppm REAL,
                gps_lat REAL,
                gps_lon REAL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                head_type TEXT NOT NULL,
                detection_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                label TEXT,
                image_path TEXT,
                gps_lat REAL,
                gps_lon REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS mission_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                mission_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT,
                gps_lat REAL,
                gps_lon REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sensor_type ON sensor_readings(sensor_type);
            CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detection_type ON detections(detection_type);
            CREATE INDEX IF NOT EXISTS idx_mission_id ON mission_logs(mission_id);
            CREATE INDEX IF NOT EXISTS idx_mission_event ON mission_logs(event_type);
            CREATE INDEX IF NOT EXISTS idx_system_event_type ON system_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_system_severity ON system_events(severity);
        """
        try:
            conn: sqlite3.Connection = self._get_connection()
            conn.executescript(schema)
            conn.commit()
            logger.info("Database schema verified")
        except sqlite3.Error as e:
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    def insert(self, table: str, data: dict[str, Any]) -> int:
        """
        Insert a row into the specified table.
        Returns the row ID of the inserted record.
        """
        if not data:
            raise DatabaseError("Cannot insert empty data")

        columns: str = ", ".join(data.keys())
        placeholders: str = ", ".join(["?"] * len(data))
        sql: str = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self._transaction() as conn:
            cursor: sqlite3.Cursor = conn.execute(sql, list(data.values()))
            row_id: int = cursor.lastrowid or 0
            logger.debug("Inserted row %d into %s", row_id, table)
            return row_id

    def insert_many(self, table: str, rows: list[dict[str, Any]]) -> int:
        """
        Bulk insert multiple rows. All rows must have the same columns.
        Returns the number of rows inserted.
        """
        if not rows:
            return 0

        columns: str = ", ".join(rows[0].keys())
        placeholders: str = ", ".join(["?"] * len(rows[0]))
        sql: str = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self._transaction() as conn:
            conn.executemany(sql, [list(row.values()) for row in rows])
            count: int = len(rows)
            logger.debug("Bulk inserted %d rows into %s", count, table)
            return count

    def query(
        self,
        sql: str,
        params: Optional[tuple[Any, ...]] = None,
        limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Execute a SELECT query and return results as a list of dicts.
        """
        if limit is not None:
            sql = f"{sql} LIMIT {limit}"

        try:
            conn: sqlite3.Connection = self._get_connection()
            cursor: sqlite3.Cursor = conn.execute(sql, params or ())
            rows: list[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Query failed: {e}") from e

    def query_one(
        self,
        sql: str,
        params: Optional[tuple[Any, ...]] = None
    ) -> Optional[dict[str, Any]]:
        """Execute a SELECT query and return the first result or None."""
        results: list[dict[str, Any]] = self.query(sql, params, limit=1)
        return results[0] if results else None

    def execute(self, sql: str, params: Optional[tuple[Any, ...]] = None) -> int:
        """
        Execute a non-SELECT statement (UPDATE, DELETE).
        Returns the number of affected rows.
        """
        with self._transaction() as conn:
            cursor: sqlite3.Cursor = conn.execute(sql, params or ())
            affected: int = cursor.rowcount
            logger.debug("Executed SQL: %d rows affected", affected)
            return affected

    def log_health(self, data: dict[str, Any]) -> int:
        """Convenience: insert a health snapshot."""
        return self.insert("health_snapshots", data)

    def log_sensor(self, data: dict[str, Any]) -> int:
        """Convenience: insert a sensor reading."""
        return self.insert("sensor_readings", data)

    def log_detection(self, data: dict[str, Any]) -> int:
        """Convenience: insert an AI detection."""
        return self.insert("detections", data)

    def log_mission_event(self, data: dict[str, Any]) -> int:
        """Convenience: insert a mission log entry."""
        return self.insert("mission_logs", data)

    def log_system_event(
        self,
        event_type: str,
        source: str,
        message: str,
        severity: str = "INFO",
        metadata: Optional[str] = None
    ) -> int:
        """Convenience: insert a system event."""
        return self.insert("system_events", {
            "event_type": event_type,
            "source": source,
            "message": message,
            "severity": severity,
            "metadata": metadata
        })

    def get_latest_health(self) -> Optional[dict[str, Any]]:
        """Get the most recent health snapshot."""
        return self.query_one(
            "SELECT * FROM health_snapshots ORDER BY timestamp DESC"
        )

    def get_detections_since(self, since: str) -> list[dict[str, Any]]:
        """Get all detections since a given ISO timestamp."""
        return self.query(
            "SELECT * FROM detections WHERE timestamp > ? ORDER BY timestamp DESC",
            (since,)
        )

    def get_table_count(self, table: str) -> int:
        """Get the row count for a table."""
        result: Optional[dict[str, Any]] = self.query_one(
            f"SELECT COUNT(*) as count FROM {table}"
        )
        return result["count"] if result else 0

    def purge_old_records(self, table: str, days: int = 30) -> int:
        """Delete records older than the specified number of days."""
        return self.execute(
            f"DELETE FROM {table} WHERE timestamp < datetime('now', '-{days} days')"
        )

    def close(self) -> None:
        """Close the current thread's database connection."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
                self._local.connection = None
                logger.debug("Database connection closed for thread %s", threading.current_thread().name)
            except sqlite3.Error as e:
                logger.error("Error closing database connection: %s", e)

    def close_all(self) -> None:
        """Close current thread connection. Called during shutdown."""
        self.close()
        logger.info("Database shutdown complete")

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing. Not for production use."""
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False

    def __repr__(self) -> str:
        return f"CerberusDB(path='{self._db_path}')"