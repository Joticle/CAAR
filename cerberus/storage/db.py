"""
Cerberus SQLite Database Manager
Autonomous edge storage for all rover data: health snapshots, sensor readings,
AI detections, species sightings, pest behavior, microclimate grids, navigation
logs, mission events, occupancy grid, and system events. WAL mode for concurrent
read/write. Thread-safe with connection-per-thread pattern.
"""

import json
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
        self._max_log_age_days: int = config.get("database", "max_log_age_days", default=30)
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
            -- =============================================================
            -- HEALTH MONITORING
            -- =============================================================
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
                battery_temp_c REAL,
                motor_current_a REAL,
                accessory_current_a REAL,
                gps_lat REAL,
                gps_lon REAL,
                gps_fix INTEGER DEFAULT 0,
                wifi_signal_dbm INTEGER,
                uptime_seconds REAL
            );

            -- =============================================================
            -- ENVIRONMENTAL SENSOR READINGS
            -- =============================================================
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                sensor_type TEXT NOT NULL,
                temperature_c REAL,
                humidity_pct REAL,
                pressure_hpa REAL,
                gas_resistance_ohms REAL,
                voc_index REAL,
                co2_ppm REAL,
                heat_index_c REAL,
                dew_point_c REAL,
                comfort_level TEXT,
                gps_lat REAL,
                gps_lon REAL
            );

            -- =============================================================
            -- AI DETECTIONS (all heads)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                head_type TEXT NOT NULL,
                detection_type TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT,
                gps_lat REAL,
                gps_lon REAL,
                gps_alt_m REAL,
                inference_time_ms REAL,
                metadata TEXT
            );

            -- =============================================================
            -- SPECIES SIGHTINGS (bird watcher)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS species_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                species TEXT NOT NULL,
                common_name TEXT,
                scientific_name TEXT,
                identification_method TEXT NOT NULL DEFAULT 'visual',
                confidence REAL NOT NULL,
                photo_path TEXT,
                audio_path TEXT,
                gps_lat REAL,
                gps_lon REAL,
                gps_alt_m REAL,
                notes TEXT
            );

            -- =============================================================
            -- SPECIES CATALOG (aggregated bird/wildlife stats)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS species_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL UNIQUE,
                common_name TEXT,
                scientific_name TEXT,
                category TEXT NOT NULL DEFAULT 'unknown',
                sighting_count INTEGER NOT NULL DEFAULT 0,
                best_confidence REAL DEFAULT 0.0,
                best_photo_path TEXT,
                first_seen TEXT,
                last_seen TEXT
            );

            -- =============================================================
            -- PEST BEHAVIOR (adaptive deterrent learning)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS pest_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL UNIQUE,
                encounters INTEGER NOT NULL DEFAULT 0,
                deterred_count INTEGER NOT NULL DEFAULT 0,
                ignored_count INTEGER NOT NULL DEFAULT 0,
                effective_actions TEXT,
                current_escalation TEXT NOT NULL DEFAULT 'gentle',
                deterrent_rate REAL DEFAULT 0.0,
                last_seen TEXT,
                last_effective_action TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- =============================================================
            -- PEST EVENTS (individual deterrent actions)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS pest_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                species TEXT NOT NULL,
                confidence REAL,
                deterrent_action TEXT NOT NULL,
                escalation TEXT NOT NULL,
                response_effective INTEGER DEFAULT 0,
                gps_lat REAL,
                gps_lon REAL,
                image_path TEXT
            );

            -- =============================================================
            -- WEED DETECTIONS (geotag + classification)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS weed_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                species TEXT NOT NULL DEFAULT 'unknown_weed',
                confidence REAL NOT NULL,
                image_path TEXT,
                gps_lat REAL NOT NULL,
                gps_lon REAL NOT NULL,
                gps_hdop REAL,
                scan_position_pan REAL,
                scan_position_tilt REAL,
                grid_row INTEGER,
                grid_col INTEGER,
                verified INTEGER DEFAULT 0
            );

            -- =============================================================
            -- SURVEILLANCE EVENTS (motion + threat)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS surveillance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                threat_level TEXT NOT NULL DEFAULT 'none',
                label TEXT,
                confidence REAL,
                region_count INTEGER DEFAULT 0,
                motion_pct REAL DEFAULT 0.0,
                image_path TEXT,
                gps_lat REAL,
                gps_lon REAL
            );

            -- =============================================================
            -- MICROCLIMATE READINGS (grid-point measurements)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS microclimate_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                survey_name TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                humidity_pct REAL NOT NULL,
                ground_temp_c REAL,
                ambient_temp_c REAL,
                temp_differential_c REAL,
                heat_index_c REAL,
                gps_lat REAL NOT NULL,
                gps_lon REAL NOT NULL,
                grid_row INTEGER NOT NULL,
                grid_col INTEGER NOT NULL,
                probe_height_cm REAL
            );

            -- =============================================================
            -- MICROCLIMATE SURVEYS (completed survey summaries)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS microclimate_surveys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                survey_name TEXT NOT NULL UNIQUE,
                total_readings INTEGER NOT NULL,
                duration_seconds REAL,
                grid_rows INTEGER,
                grid_cols INTEGER,
                temp_min REAL,
                temp_max REAL,
                temp_avg REAL,
                temp_stdev REAL,
                humidity_min REAL,
                humidity_max REAL,
                humidity_avg REAL,
                hotspot_count INTEGER DEFAULT 0,
                coldspot_count INTEGER DEFAULT 0,
                moisture_zone_count INTEGER DEFAULT 0,
                heatmap_data TEXT
            );

            -- =============================================================
            -- NAVIGATION LOG (waypoint arrivals, route events)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS navigation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                target_lat REAL,
                target_lon REAL,
                actual_lat REAL,
                actual_lon REAL,
                distance_m REAL,
                heading_deg REAL,
                speed REAL,
                waypoint_name TEXT,
                status TEXT,
                message TEXT
            );

            -- =============================================================
            -- MISSION EVENTS (lifecycle + task tracking)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS mission_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                mission_id TEXT NOT NULL,
                mission_name TEXT,
                event_type TEXT NOT NULL,
                task_id TEXT,
                task_type TEXT,
                head_type TEXT,
                gps_lat REAL,
                gps_lon REAL,
                message TEXT,
                metadata TEXT
            );

            -- =============================================================
            -- PATROL LOG (route execution tracking)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS patrol_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                route_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                waypoint_name TEXT,
                waypoint_index INTEGER,
                loop_number INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                action TEXT,
                dwell_seconds REAL,
                message TEXT
            );

            -- =============================================================
            -- OCCUPANCY GRID (path planning obstacle map)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS occupancy_grid (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_x INTEGER NOT NULL,
                grid_y INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL DEFAULT 0.0,
                last_observed TEXT NOT NULL DEFAULT (datetime('now')),
                observation_count INTEGER DEFAULT 1,
                gps_lat REAL,
                gps_lon REAL,
                UNIQUE(grid_x, grid_y)
            );

            -- =============================================================
            -- RTB EVENTS (return to base history)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS rtb_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                reason TEXT NOT NULL,
                trigger_value TEXT,
                start_lat REAL,
                start_lon REAL,
                home_lat REAL,
                home_lon REAL,
                distance_m REAL,
                result TEXT NOT NULL,
                duration_seconds REAL,
                retries INTEGER DEFAULT 0,
                message TEXT
            );

            -- =============================================================
            -- SYSTEM EVENTS (boot, shutdown, errors, safety)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT
            );

            -- =============================================================
            -- CONFIDENCE TRACKING (model performance calibration)
            -- =============================================================
            CREATE TABLE IF NOT EXISTS confidence_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                model_name TEXT NOT NULL,
                label TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                verified INTEGER DEFAULT 0,
                correct INTEGER,
                notes TEXT
            );

            -- =============================================================
            -- INDEXES
            -- =============================================================
            CREATE INDEX IF NOT EXISTS idx_health_ts ON health_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sensor_ts ON sensor_readings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sensor_type ON sensor_readings(sensor_type);
            CREATE INDEX IF NOT EXISTS idx_detect_ts ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detect_type ON detections(detection_type);
            CREATE INDEX IF NOT EXISTS idx_detect_head ON detections(head_type);
            CREATE INDEX IF NOT EXISTS idx_detect_label ON detections(label);
            CREATE INDEX IF NOT EXISTS idx_sighting_ts ON species_sightings(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sighting_species ON species_sightings(species);
            CREATE INDEX IF NOT EXISTS idx_sighting_method ON species_sightings(identification_method);
            CREATE INDEX IF NOT EXISTS idx_pest_species ON pest_behavior(species);
            CREATE INDEX IF NOT EXISTS idx_pest_event_ts ON pest_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pest_event_species ON pest_events(species);
            CREATE INDEX IF NOT EXISTS idx_weed_ts ON weed_detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_weed_gps ON weed_detections(gps_lat, gps_lon);
            CREATE INDEX IF NOT EXISTS idx_weed_verified ON weed_detections(verified);
            CREATE INDEX IF NOT EXISTS idx_surv_ts ON surveillance_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_surv_threat ON surveillance_events(threat_level);
            CREATE INDEX IF NOT EXISTS idx_micro_survey ON microclimate_readings(survey_name);
            CREATE INDEX IF NOT EXISTS idx_micro_grid ON microclimate_readings(grid_row, grid_col);
            CREATE INDEX IF NOT EXISTS idx_nav_ts ON navigation_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_nav_type ON navigation_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_mission_id ON mission_events(mission_id);
            CREATE INDEX IF NOT EXISTS idx_mission_type ON mission_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_patrol_route ON patrol_log(route_name);
            CREATE INDEX IF NOT EXISTS idx_patrol_ts ON patrol_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_occgrid_pos ON occupancy_grid(grid_x, grid_y);
            CREATE INDEX IF NOT EXISTS idx_occgrid_state ON occupancy_grid(state);
            CREATE INDEX IF NOT EXISTS idx_rtb_ts ON rtb_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_rtb_reason ON rtb_events(reason);
            CREATE INDEX IF NOT EXISTS idx_sys_ts ON system_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_sys_type ON system_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_sys_severity ON system_events(severity);
            CREATE INDEX IF NOT EXISTS idx_conf_model ON confidence_tracking(model_name);
            CREATE INDEX IF NOT EXISTS idx_conf_label ON confidence_tracking(label);
        """
        try:
            conn: sqlite3.Connection = self._get_connection()
            conn.executescript(schema)
            conn.commit()
            logger.info("Database schema verified — all tables ready")
        except sqlite3.Error as e:
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    # =================================================================
    # GENERIC OPERATIONS
    # =================================================================

    def insert(self, table: str, data: dict[str, Any]) -> int:
        """Insert a row. Returns the row ID."""
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
        """Bulk insert. All rows must have the same columns. Returns count."""
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

    def upsert(self, table: str, data: dict[str, Any], conflict_columns: list[str]) -> int:
        """Insert or update on conflict. Returns row ID."""
        if not data:
            raise DatabaseError("Cannot upsert empty data")

        columns: str = ", ".join(data.keys())
        placeholders: str = ", ".join(["?"] * len(data))
        conflict: str = ", ".join(conflict_columns)
        updates: str = ", ".join(
            f"{k} = excluded.{k}" for k in data.keys() if k not in conflict_columns
        )
        sql: str = (
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        )

        with self._transaction() as conn:
            cursor: sqlite3.Cursor = conn.execute(sql, list(data.values()))
            row_id: int = cursor.lastrowid or 0
            return row_id

    def query(
        self,
        sql: str,
        params: Optional[tuple[Any, ...]] = None,
        limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query. Returns list of dicts."""
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
        """Execute a SELECT query. Returns first result or None."""
        results: list[dict[str, Any]] = self.query(sql, params, limit=1)
        return results[0] if results else None

    def execute(self, sql: str, params: Optional[tuple[Any, ...]] = None) -> int:
        """Execute a non-SELECT statement. Returns affected row count."""
        with self._transaction() as conn:
            cursor: sqlite3.Cursor = conn.execute(sql, params or ())
            affected: int = cursor.rowcount
            logger.debug("Executed SQL: %d rows affected", affected)
            return affected

    def get_table_count(self, table: str) -> int:
        """Get row count for a table."""
        result: Optional[dict[str, Any]] = self.query_one(
            f"SELECT COUNT(*) as count FROM {table}"
        )
        return result["count"] if result else 0

    # =================================================================
    # HEALTH
    # =================================================================

    def log_health(self, data: dict[str, Any]) -> int:
        """Insert a health snapshot."""
        return self.insert("health_snapshots", data)

    def get_latest_health(self) -> Optional[dict[str, Any]]:
        """Get the most recent health snapshot."""
        return self.query_one(
            "SELECT * FROM health_snapshots ORDER BY timestamp DESC"
        )

    def get_health_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get health snapshots for the last N hours."""
        return self.query(
            "SELECT * FROM health_snapshots WHERE timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # SENSOR READINGS
    # =================================================================

    def log_sensor(self, data: dict[str, Any]) -> int:
        """Insert a sensor reading."""
        return self.insert("sensor_readings", data)

    def log_sensor_reading(self, sensor_type: str, data: dict[str, Any]) -> int:
        """Insert a sensor reading with type. Flattens nested data to metadata."""
        row: dict[str, Any] = {"sensor_type": sensor_type}
        known_cols: set[str] = {
            "temperature_c", "humidity_pct", "pressure_hpa", "gas_resistance_ohms",
            "voc_index", "co2_ppm", "heat_index_c", "dew_point_c", "comfort_level",
            "gps_lat", "gps_lon"
        }
        for k, v in data.items():
            if k in known_cols:
                row[k] = v
        return self.insert("sensor_readings", row)

    def get_sensor_readings(
        self, sensor_type: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get sensor readings by type for the last N hours."""
        return self.query(
            "SELECT * FROM sensor_readings WHERE sensor_type = ? AND timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (sensor_type, f"-{hours} hours")
        )

    # =================================================================
    # AI DETECTIONS
    # =================================================================

    def log_detection(self, data: dict[str, Any]) -> int:
        """Insert an AI detection."""
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = json.dumps(data["metadata"])
        return self.insert("detections", data)

    def get_detections_since(self, since: str) -> list[dict[str, Any]]:
        """Get all detections since a given ISO timestamp."""
        return self.query(
            "SELECT * FROM detections WHERE timestamp > ? ORDER BY timestamp DESC",
            (since,)
        )

    def get_detections_by_type(
        self, detection_type: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get detections by type for the last N hours."""
        return self.query(
            "SELECT * FROM detections WHERE detection_type = ? AND timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (detection_type, f"-{hours} hours")
        )

    def get_detection_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get detection counts grouped by type and label."""
        return self.query(
            "SELECT detection_type, label, COUNT(*) as count, AVG(confidence) as avg_confidence "
            "FROM detections WHERE timestamp > datetime('now', ?) "
            "GROUP BY detection_type, label ORDER BY count DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # SPECIES SIGHTINGS (Bird Watcher)
    # =================================================================

    def log_sighting(self, data: dict[str, Any]) -> int:
        """Insert a species sighting."""
        row_id: int = self.insert("species_sightings", data)
        self._update_species_catalog(data)
        return row_id

    def _update_species_catalog(self, sighting: dict[str, Any]) -> None:
        """Update the species catalog with new sighting data."""
        species: str = sighting.get("species", "unknown")
        existing: Optional[dict[str, Any]] = self.query_one(
            "SELECT * FROM species_catalog WHERE species = ?", (species,)
        )

        if existing is None:
            self.insert("species_catalog", {
                "species": species,
                "common_name": sighting.get("common_name", species),
                "scientific_name": sighting.get("scientific_name", ""),
                "category": sighting.get("category", "bird"),
                "sighting_count": 1,
                "best_confidence": sighting.get("confidence", 0.0),
                "best_photo_path": sighting.get("photo_path", ""),
                "first_seen": sighting.get("timestamp", ""),
                "last_seen": sighting.get("timestamp", "")
            })
        else:
            updates: dict[str, Any] = {
                "sighting_count": existing["sighting_count"] + 1,
                "last_seen": sighting.get("timestamp", "")
            }
            confidence: float = sighting.get("confidence", 0.0)
            if confidence > (existing["best_confidence"] or 0.0):
                updates["best_confidence"] = confidence
                if sighting.get("photo_path"):
                    updates["best_photo_path"] = sighting["photo_path"]
            if sighting.get("scientific_name") and not existing.get("scientific_name"):
                updates["scientific_name"] = sighting["scientific_name"]

            set_clause: str = ", ".join(f"{k} = ?" for k in updates.keys())
            self.execute(
                f"UPDATE species_catalog SET {set_clause} WHERE species = ?",
                (*updates.values(), species)
            )

    def get_species_catalog(self) -> list[dict[str, Any]]:
        """Get the full species catalog sorted by sighting count."""
        return self.query(
            "SELECT * FROM species_catalog ORDER BY sighting_count DESC"
        )

    def get_recent_sightings(self, count: int = 20) -> list[dict[str, Any]]:
        """Get the most recent sightings."""
        return self.query(
            "SELECT * FROM species_sightings ORDER BY timestamp DESC",
            limit=count
        )

    # =================================================================
    # PEST BEHAVIOR (Adaptive Learning)
    # =================================================================

    def update_pest_behavior(self, data: dict[str, Any]) -> int:
        """Upsert pest behavior record for a species."""
        if "effective_actions" in data and isinstance(data["effective_actions"], dict):
            data["effective_actions"] = json.dumps(data["effective_actions"])
        data["updated_at"] = "datetime('now')"
        return self.upsert("pest_behavior", data, ["species"])

    def log_pest_event(self, data: dict[str, Any]) -> int:
        """Insert a pest deterrent event."""
        return self.insert("pest_events", data)

    def get_pest_behavior(self, species: str) -> Optional[dict[str, Any]]:
        """Get behavior profile for a specific pest species."""
        result: Optional[dict[str, Any]] = self.query_one(
            "SELECT * FROM pest_behavior WHERE species = ?", (species,)
        )
        if result and result.get("effective_actions"):
            try:
                result["effective_actions"] = json.loads(result["effective_actions"])
            except (json.JSONDecodeError, TypeError):
                result["effective_actions"] = {}
        return result

    def get_all_pest_behaviors(self) -> list[dict[str, Any]]:
        """Get all pest behavior profiles."""
        results: list[dict[str, Any]] = self.query(
            "SELECT * FROM pest_behavior ORDER BY encounters DESC"
        )
        for r in results:
            if r.get("effective_actions"):
                try:
                    r["effective_actions"] = json.loads(r["effective_actions"])
                except (json.JSONDecodeError, TypeError):
                    r["effective_actions"] = {}
        return results

    def get_pest_activity(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get pest events for the last N hours."""
        return self.query(
            "SELECT species, COUNT(*) as events, SUM(response_effective) as effective, "
            "AVG(confidence) as avg_confidence FROM pest_events "
            "WHERE timestamp > datetime('now', ?) GROUP BY species ORDER BY events DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # WEED DETECTIONS
    # =================================================================

    def log_weed_detection(self, data: dict[str, Any]) -> int:
        """Insert a weed detection with geotag."""
        return self.insert("weed_detections", data)

    def get_weed_hotspots(self, min_detections: int = 3) -> list[dict[str, Any]]:
        """Get locations with recurring weed detections."""
        return self.query(
            "SELECT ROUND(gps_lat, 5) as lat_zone, ROUND(gps_lon, 5) as lon_zone, "
            "COUNT(*) as detection_count, AVG(confidence) as avg_confidence, "
            "GROUP_CONCAT(DISTINCT species) as species_found "
            "FROM weed_detections GROUP BY lat_zone, lon_zone "
            "HAVING detection_count >= ? ORDER BY detection_count DESC",
            (min_detections,)
        )

    def get_unverified_weeds(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get weed detections pending verification."""
        return self.query(
            "SELECT * FROM weed_detections WHERE verified = 0 ORDER BY confidence DESC",
            limit=limit
        )

    # =================================================================
    # SURVEILLANCE EVENTS
    # =================================================================

    def log_surveillance_event(self, data: dict[str, Any]) -> int:
        """Insert a surveillance event."""
        return self.insert("surveillance_events", data)

    def get_threat_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get surveillance events with medium+ threat level."""
        return self.query(
            "SELECT * FROM surveillance_events WHERE threat_level IN ('medium', 'high', 'critical') "
            "AND timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (f"-{hours} hours",)
        )

    def get_motion_activity(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get motion activity summary by hour."""
        return self.query(
            "SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour, COUNT(*) as events, "
            "AVG(motion_pct) as avg_motion_pct, MAX(threat_level) as max_threat "
            "FROM surveillance_events WHERE timestamp > datetime('now', ?) "
            "GROUP BY hour ORDER BY hour ASC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # MICROCLIMATE
    # =================================================================

    def log_microclimate_reading(self, data: dict[str, Any]) -> int:
        """Insert a microclimate grid point reading."""
        return self.insert("microclimate_readings", data)

    def log_microclimate_survey(self, data: dict[str, Any]) -> int:
        """Insert or update a completed survey summary."""
        if "heatmap_data" in data and isinstance(data["heatmap_data"], dict):
            data["heatmap_data"] = json.dumps(data["heatmap_data"])
        return self.upsert("microclimate_surveys", data, ["survey_name"])

    def get_survey_readings(self, survey_name: str) -> list[dict[str, Any]]:
        """Get all readings for a specific survey."""
        return self.query(
            "SELECT * FROM microclimate_readings WHERE survey_name = ? ORDER BY grid_row, grid_col",
            (survey_name,)
        )

    def get_survey_list(self) -> list[dict[str, Any]]:
        """Get summary of all completed surveys."""
        return self.query(
            "SELECT survey_name, total_readings, duration_seconds, temp_min, temp_max, "
            "temp_avg, hotspot_count, coldspot_count, timestamp "
            "FROM microclimate_surveys ORDER BY timestamp DESC"
        )

    # =================================================================
    # NAVIGATION
    # =================================================================

    def log_navigation_event(self, data: dict[str, Any]) -> int:
        """Insert a navigation event."""
        return self.insert("navigation_log", data)

    def get_navigation_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get navigation events for the last N hours."""
        return self.query(
            "SELECT * FROM navigation_log WHERE timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # MISSIONS
    # =================================================================

    def log_mission_event(self, data: dict[str, Any]) -> int:
        """Insert a mission lifecycle event."""
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = json.dumps(data["metadata"])
        return self.insert("mission_events", data)

    def get_mission_history(
        self, mission_id: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get mission events, optionally filtered by mission ID."""
        if mission_id:
            return self.query(
                "SELECT * FROM mission_events WHERE mission_id = ? ORDER BY timestamp ASC",
                (mission_id,), limit=limit
            )
        return self.query(
            "SELECT * FROM mission_events ORDER BY timestamp DESC", limit=limit
        )

    # =================================================================
    # PATROL
    # =================================================================

    def log_patrol_event(self, data: dict[str, Any]) -> int:
        """Insert a patrol log entry."""
        return self.insert("patrol_log", data)

    def get_patrol_history(self, route_name: Optional[str] = None, hours: int = 24) -> list[dict[str, Any]]:
        """Get patrol events, optionally filtered by route."""
        if route_name:
            return self.query(
                "SELECT * FROM patrol_log WHERE route_name = ? AND timestamp > datetime('now', ?) ORDER BY timestamp ASC",
                (route_name, f"-{hours} hours")
            )
        return self.query(
            "SELECT * FROM patrol_log WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # OCCUPANCY GRID (Path Planning)
    # =================================================================

    def update_grid_cell(self, grid_x: int, grid_y: int, state: str,
                         confidence: float = 1.0, gps_lat: float = 0.0,
                         gps_lon: float = 0.0) -> int:
        """Insert or update an occupancy grid cell."""
        return self.upsert("occupancy_grid", {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "state": state,
            "confidence": confidence,
            "last_observed": "datetime('now')",
            "gps_lat": gps_lat,
            "gps_lon": gps_lon
        }, ["grid_x", "grid_y"])

    def increment_grid_observation(self, grid_x: int, grid_y: int) -> int:
        """Increment observation count for a grid cell."""
        return self.execute(
            "UPDATE occupancy_grid SET observation_count = observation_count + 1, "
            "last_observed = datetime('now') WHERE grid_x = ? AND grid_y = ?",
            (grid_x, grid_y)
        )

    def get_occupancy_grid(self) -> list[dict[str, Any]]:
        """Get the full occupancy grid."""
        return self.query(
            "SELECT * FROM occupancy_grid ORDER BY grid_x, grid_y"
        )

    def get_obstacles(self) -> list[dict[str, Any]]:
        """Get all cells marked as occupied/obstacle."""
        return self.query(
            "SELECT * FROM occupancy_grid WHERE state = 'occupied' ORDER BY last_observed DESC"
        )

    def clear_stale_obstacles(self, days: int = 7) -> int:
        """Clear obstacles not observed recently (environment may have changed)."""
        return self.execute(
            "UPDATE occupancy_grid SET state = 'unknown', confidence = 0.0 "
            "WHERE state = 'occupied' AND last_observed < datetime('now', ?)",
            (f"-{days} days",)
        )

    # =================================================================
    # RTB EVENTS
    # =================================================================

    def log_rtb_event(self, data: dict[str, Any]) -> int:
        """Insert a return-to-base event."""
        return self.insert("rtb_events", data)

    def get_rtb_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent RTB events."""
        return self.query(
            "SELECT * FROM rtb_events ORDER BY timestamp DESC", limit=limit
        )

    # =================================================================
    # SYSTEM EVENTS
    # =================================================================

    def log_system_event(
        self,
        event_type: str,
        source: str,
        message: str,
        severity: str = "INFO",
        metadata: Optional[str] = None
    ) -> int:
        """Insert a system event."""
        if metadata is not None and isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        return self.insert("system_events", {
            "event_type": event_type,
            "source": source,
            "message": message,
            "severity": severity,
            "metadata": metadata
        })

    def get_system_events(
        self, severity: Optional[str] = None, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get system events, optionally filtered by severity."""
        if severity:
            return self.query(
                "SELECT * FROM system_events WHERE severity = ? AND timestamp > datetime('now', ?) ORDER BY timestamp DESC",
                (severity, f"-{hours} hours")
            )
        return self.query(
            "SELECT * FROM system_events WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # CONFIDENCE TRACKING (Model Calibration)
    # =================================================================

    def log_confidence(self, model_name: str, label: str, confidence: float) -> int:
        """Log a model prediction for later verification."""
        return self.insert("confidence_tracking", {
            "model_name": model_name,
            "label": label,
            "predicted_confidence": confidence
        })

    def verify_prediction(self, tracking_id: int, correct: bool, notes: str = "") -> int:
        """Mark a prediction as verified correct or incorrect."""
        return self.execute(
            "UPDATE confidence_tracking SET verified = 1, correct = ?, notes = ? WHERE id = ?",
            (1 if correct else 0, notes, tracking_id)
        )

    def get_model_accuracy(self, model_name: str) -> Optional[dict[str, Any]]:
        """Get accuracy stats for a model based on verified predictions."""
        return self.query_one(
            "SELECT model_name, COUNT(*) as total_verified, "
            "SUM(correct) as correct_count, "
            "ROUND(CAST(SUM(correct) AS REAL) / COUNT(*) * 100, 1) as accuracy_pct, "
            "AVG(predicted_confidence) as avg_confidence "
            "FROM confidence_tracking WHERE model_name = ? AND verified = 1",
            (model_name,)
        )

    def get_false_positive_rate(self, model_name: str, label: str) -> Optional[dict[str, Any]]:
        """Get false positive rate for a specific model + label combo."""
        return self.query_one(
            "SELECT label, COUNT(*) as total, SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as false_positives, "
            "ROUND(CAST(SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100, 1) as fp_rate_pct "
            "FROM confidence_tracking WHERE model_name = ? AND label = ? AND verified = 1",
            (model_name, label)
        )

    # =================================================================
    # MAINTENANCE
    # =================================================================

    def purge_old_records(self, table: str, days: Optional[int] = None) -> int:
        """Delete records older than the specified number of days."""
        if days is None:
            days = self._max_log_age_days
        return self.execute(
            f"DELETE FROM {table} WHERE timestamp < datetime('now', '-{days} days')"
        )

    def purge_all_old_records(self, days: Optional[int] = None) -> dict[str, int]:
        """Purge old records from all time-series tables. Returns counts per table."""
        tables: list[str] = [
            "health_snapshots", "sensor_readings", "detections",
            "species_sightings", "pest_events", "weed_detections",
            "surveillance_events", "microclimate_readings", "navigation_log",
            "mission_events", "patrol_log", "rtb_events",
            "system_events", "confidence_tracking"
        ]
        results: dict[str, int] = {}
        for table in tables:
            try:
                results[table] = self.purge_old_records(table, days)
            except DatabaseError as e:
                logger.error("Failed to purge %s: %s", table, e)
                results[table] = -1
        total: int = sum(v for v in results.values() if v > 0)
        logger.info("Purged %d total old records across %d tables", total, len(tables))
        return results

    def get_database_stats(self) -> dict[str, Any]:
        """Get row counts and database file size for all tables."""
        tables: list[str] = [
            "health_snapshots", "sensor_readings", "detections",
            "species_sightings", "species_catalog", "pest_behavior",
            "pest_events", "weed_detections", "surveillance_events",
            "microclimate_readings", "microclimate_surveys",
            "navigation_log", "mission_events", "patrol_log",
            "occupancy_grid", "rtb_events", "system_events",
            "confidence_tracking"
        ]
        stats: dict[str, Any] = {"tables": {}}
        for table in tables:
            try:
                stats["tables"][table] = self.get_table_count(table)
            except DatabaseError:
                stats["tables"][table] = -1

        try:
            stats["file_size_mb"] = round(os.path.getsize(self._db_path) / (1024 * 1024), 2)
        except OSError:
            stats["file_size_mb"] = 0.0

        stats["total_rows"] = sum(v for v in stats["tables"].values() if v > 0)
        return stats

    def vacuum(self) -> None:
        """Reclaim disk space. Run during maintenance windows only."""
        try:
            conn: sqlite3.Connection = self._get_connection()
            conn.execute("VACUUM")
            logger.info("Database vacuum complete")
        except sqlite3.Error as e:
            logger.error("Vacuum failed: %s", e)

    # =================================================================
    # CONNECTION MANAGEMENT
    # =================================================================

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
