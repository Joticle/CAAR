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
        db_dir: str = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is None:
            try:
                conn = sqlite3.connect(self._db_path, timeout=self._busy_timeout / 1000)
                conn.row_factory = sqlite3.Row
                conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout)}")
                conn.execute(f"PRAGMA journal_mode = {self._journal_mode}")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA synchronous = NORMAL")
                self._local.connection = conn
            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to connect to database: {e}") from e
        return conn

    @contextmanager
    def _transaction(self):
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

    @contextmanager
    def transaction(self):
        """Public transaction context manager."""
        with self._transaction() as conn:
            yield conn

    def _init_schema(self) -> None:
        schema: str = """
            CREATE TABLE IF NOT EXISTS health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                cpu_temp_c REAL,
                cpu_usage_pct REAL,
                cpu_pct REAL,
                memory_usage_pct REAL,
                memory_pct REAL,
                disk_usage_pct REAL,
                disk_pct REAL,
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

            CREATE TABLE IF NOT EXISTS weed_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                species TEXT NOT NULL DEFAULT 'unknown_weed',
                confidence REAL NOT NULL,
                image_path TEXT,
                gps_lat REAL NOT NULL,
                gps_lon REAL NOT NULL,
                gps_hdop REAL,
                hdop REAL,
                scan_position_pan REAL,
                scan_position_tilt REAL,
                scan_pan REAL,
                scan_tilt REAL,
                grid_row INTEGER,
                grid_col INTEGER,
                verified INTEGER DEFAULT 0
            );

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
                temp_min_c REAL,
                temp_max_c REAL,
                temp_avg REAL,
                temp_avg_c REAL,
                temp_stdev REAL,
                temp_stdev_c REAL,
                humidity_min REAL,
                humidity_max REAL,
                humidity_min_pct REAL,
                humidity_max_pct REAL,
                humidity_avg REAL,
                humidity_avg_pct REAL,
                hotspot_count INTEGER DEFAULT 0,
                coldspot_count INTEGER DEFAULT 0,
                moisture_zone_count INTEGER DEFAULT 0,
                heatmap_data TEXT
            );

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

            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata TEXT
            );

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

    def query(self, sql: str, params: Optional[tuple[Any, ...]] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
        if limit is not None:
            sql = f"{sql} LIMIT {limit}"
        try:
            conn: sqlite3.Connection = self._get_connection()
            cursor: sqlite3.Cursor = conn.execute(sql, params or ())
            rows: list[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"Query failed: {e}") from e

    def query_one(self, sql: str, params: Optional[tuple[Any, ...]] = None) -> Optional[dict[str, Any]]:
        results: list[dict[str, Any]] = self.query(sql, params, limit=1)
        return results[0] if results else None

    def execute(self, sql: str, params: Optional[tuple[Any, ...]] = None) -> int:
        with self._transaction() as conn:
            cursor: sqlite3.Cursor = conn.execute(sql, params or ())
            affected: int = cursor.rowcount
            logger.debug("Executed SQL: %d rows affected", affected)
            return affected

    def get_table_count(self, table: str) -> int:
        result: Optional[dict[str, Any]] = self.query_one(f"SELECT COUNT(*) as count FROM {table}")
        return result["count"] if result else 0

    # =================================================================
    # HEALTH
    # =================================================================

    def log_health(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("health_snapshots", data or kwargs)

    def get_latest_health(self) -> Optional[dict[str, Any]]:
        return self.query_one("SELECT * FROM health_snapshots ORDER BY timestamp DESC")

    def get_health_history(self, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM health_snapshots WHERE timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # SENSOR READINGS
    # =================================================================

    def log_sensor(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("sensor_readings", data or kwargs)

    def log_sensor_reading(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        if "sensor_type" not in row:
            row["sensor_type"] = "unknown"
        return self.insert("sensor_readings", row)

    def get_sensor_readings(self, sensor_type: str, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM sensor_readings WHERE sensor_type = ? AND timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (sensor_type, f"-{hours} hours")
        )

    # =================================================================
    # AI DETECTIONS
    # =================================================================

    def log_detection(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        if "metadata" in row and isinstance(row["metadata"], dict):
            row["metadata"] = json.dumps(row["metadata"])
        return self.insert("detections", row)

    def get_detections_since(self, since: str = None, hours: int = None) -> list[dict[str, Any]]:
        if hours is not None:
            return self.query(
                "SELECT * FROM detections WHERE timestamp > datetime('now', ?) ORDER BY timestamp DESC",
                (f"-{hours} hours",)
            )
        return self.query(
            "SELECT * FROM detections WHERE timestamp > ? ORDER BY timestamp DESC",
            (since,)
        )

    def get_detections_by_type(self, detection_type: str, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM detections WHERE detection_type = ? AND timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (detection_type, f"-{hours} hours")
        )

    def get_detection_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT detection_type, label, COUNT(*) as count, AVG(confidence) as avg_confidence "
            "FROM detections WHERE timestamp > datetime('now', ?) "
            "GROUP BY detection_type, label ORDER BY count DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # SPECIES SIGHTINGS
    # =================================================================

    def log_sighting(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        row_id: int = self.insert("species_sightings", row)
        self._update_species_catalog(row)
        return row_id

    def _update_species_catalog(self, sighting: dict[str, Any]) -> None:
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
        return self.query("SELECT * FROM species_catalog ORDER BY sighting_count DESC")

    def get_recent_sightings(self, count: int = 20, limit: int = None) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM species_sightings ORDER BY timestamp DESC",
            limit=limit or count
        )

    # =================================================================
    # PEST BEHAVIOR
    # =================================================================

    def update_pest_behavior(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        if "effective_actions" in row and isinstance(row["effective_actions"], (dict, list)):
            row["effective_actions"] = json.dumps(row["effective_actions"])
        row["updated_at"] = "datetime('now')"
        return self.upsert("pest_behavior", row, ["species"])

    def log_pest_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("pest_events", data or kwargs)

    def get_pest_behavior(self, species: str) -> Optional[dict[str, Any]]:
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
        return self.query(
            "SELECT species, COUNT(*) as events, SUM(response_effective) as effective, "
            "AVG(confidence) as avg_confidence FROM pest_events "
            "WHERE timestamp > datetime('now', ?) GROUP BY species ORDER BY events DESC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # WEED DETECTIONS
    # =================================================================

    def log_weed_detection(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("weed_detections", data or kwargs)

    def get_weed_hotspots(self, min_detections: int = 3) -> list[dict[str, Any]]:
        return self.query(
            "SELECT ROUND(gps_lat, 5) as lat_zone, ROUND(gps_lon, 5) as lon_zone, "
            "COUNT(*) as detection_count, AVG(confidence) as avg_confidence, "
            "GROUP_CONCAT(DISTINCT species) as species_found "
            "FROM weed_detections GROUP BY lat_zone, lon_zone "
            "HAVING detection_count >= ? ORDER BY detection_count DESC",
            (min_detections,)
        )

    def get_unverified_weeds(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM weed_detections WHERE verified = 0 ORDER BY confidence DESC",
            limit=limit
        )

    # =================================================================
    # SURVEILLANCE EVENTS
    # =================================================================

    def log_surveillance_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("surveillance_events", data or kwargs)

    def get_threat_history(self, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM surveillance_events WHERE threat_level IN ('medium', 'high', 'critical') "
            "AND timestamp > datetime('now', ?) ORDER BY timestamp DESC",
            (f"-{hours} hours",)
        )

    def get_motion_activity(self, hours: int = 24) -> list[dict[str, Any]]:
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

    def log_microclimate_reading(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("microclimate_readings", data or kwargs)

    def log_microclimate_survey(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        if "heatmap_data" in row and isinstance(row["heatmap_data"], dict):
            row["heatmap_data"] = json.dumps(row["heatmap_data"])
        return self.upsert("microclimate_surveys", row, ["survey_name"])

    def get_survey_readings(self, survey_name: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM microclimate_readings WHERE survey_name = ? ORDER BY grid_row, grid_col",
            (survey_name,)
        )

    def get_survey_list(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT survey_name, total_readings, duration_seconds, temp_min, temp_max, "
            "temp_avg, hotspot_count, coldspot_count, timestamp "
            "FROM microclimate_surveys ORDER BY timestamp DESC"
        )

    # =================================================================
    # NAVIGATION
    # =================================================================

    def log_navigation_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("navigation_log", data or kwargs)

    def get_navigation_history(self, hours: int = 24) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM navigation_log WHERE timestamp > datetime('now', ?) ORDER BY timestamp ASC",
            (f"-{hours} hours",)
        )

    # =================================================================
    # MISSIONS
    # =================================================================

    def log_mission_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        row: dict[str, Any] = data or kwargs
        if "metadata" in row and isinstance(row["metadata"], dict):
            row["metadata"] = json.dumps(row["metadata"])
        return self.insert("mission_events", row)

    def get_mission_history(self, mission_id: Optional[str] = None, hours: int = None, limit: int = 50) -> list[dict[str, Any]]:
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

    def log_patrol_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("patrol_log", data or kwargs)

    def get_patrol_history(self, route_name: Optional[str] = None, hours: int = 24) -> list[dict[str, Any]]:
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
    # OCCUPANCY GRID
    # =================================================================

    def update_grid_cell(self, grid_x: int = None, grid_y: int = None, state: str = "unknown",
                         confidence: float = 1.0, gps_lat: float = 0.0,
                         gps_lon: float = 0.0, **kwargs: Any) -> int:
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
        return self.execute(
            "UPDATE occupancy_grid SET observation_count = observation_count + 1, "
            "last_observed = datetime('now') WHERE grid_x = ? AND grid_y = ?",
            (grid_x, grid_y)
        )

    def get_occupancy_grid(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM occupancy_grid ORDER BY grid_x, grid_y")

    def get_obstacles(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM occupancy_grid WHERE state = 'occupied' ORDER BY last_observed DESC"
        )

    def clear_stale_obstacles(self, days: int = 7) -> int:
        return self.execute(
            "UPDATE occupancy_grid SET state = 'unknown', confidence = 0.0 "
            "WHERE state = 'occupied' AND last_observed < datetime('now', ?)",
            (f"-{days} days",)
        )

    # =================================================================
    # RTB EVENTS
    # =================================================================

    def log_rtb_event(self, data: dict[str, Any] = None, **kwargs: Any) -> int:
        return self.insert("rtb_events", data or kwargs)

    def get_rtb_history(self, hours: int = None, limit: int = 20) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM rtb_events ORDER BY timestamp DESC", limit=limit
        )

    # =================================================================
    # SYSTEM EVENTS
    # =================================================================

    def log_system_event(self, data: dict[str, Any] = None, event_type: str = None,
                         source: str = None, message: str = None,
                         severity: str = "INFO", metadata: Any = None, **kwargs: Any) -> int:
        if data is not None:
            row = data
        elif event_type is not None:
            row = {
                "event_type": event_type,
                "source": source or "",
                "message": message or "",
                "severity": severity,
                "metadata": json.dumps(metadata) if isinstance(metadata, dict) else metadata
            }
        else:
            row = kwargs
        return self.insert("system_events", row)

    def get_system_events(self, severity: Optional[str] = None, hours: int = 24) -> list[dict[str, Any]]:
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
    # CONFIDENCE TRACKING
    # =================================================================

    def log_confidence(self, data: dict[str, Any] = None, model_name: str = None,
                       label: str = None, predicted_confidence: float = None,
                       confidence: float = None, **kwargs: Any) -> int:
        if data is not None:
            row = data
        elif model_name is not None:
            row = {
                "model_name": model_name,
                "label": label or "",
                "predicted_confidence": predicted_confidence or confidence or 0.0
            }
        else:
            row = kwargs
        return self.insert("confidence_tracking", row)

    def verify_prediction(self, tracking_id: int, correct: bool, notes: str = "") -> int:
        return self.execute(
            "UPDATE confidence_tracking SET verified = 1, correct = ?, notes = ? WHERE id = ?",
            (1 if correct else 0, notes, tracking_id)
        )

    def get_model_accuracy(self, model_name: str) -> Optional[dict[str, Any]]:
        return self.query_one(
            "SELECT model_name, COUNT(*) as total_verified, "
            "SUM(correct) as correct_count, "
            "ROUND(CAST(SUM(correct) AS REAL) / COUNT(*) * 100, 1) as accuracy_pct, "
            "AVG(predicted_confidence) as avg_confidence "
            "FROM confidence_tracking WHERE model_name = ? AND verified = 1",
            (model_name,)
        )

    def get_false_positive_rate(self, model_name: str, label: str = None) -> Any:
        if label:
            result = self.query_one(
                "SELECT label, COUNT(*) as total, SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as false_positives, "
                "ROUND(CAST(SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100, 1) as fp_rate_pct "
                "FROM confidence_tracking WHERE model_name = ? AND label = ? AND verified = 1",
                (model_name, label)
            )
            return result
        result = self.query_one(
            "SELECT COUNT(*) as total, SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as false_positives, "
            "ROUND(CAST(SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*), 2) as fp_rate "
            "FROM confidence_tracking WHERE model_name = ? AND verified = 1",
            (model_name,)
        )
        return result["fp_rate"] if result else 0.0

    # =================================================================
    # MAINTENANCE
    # =================================================================

    def purge_old_records(self, table: str, days: Optional[int] = None) -> int:
        if days is None:
            days = self._max_log_age_days
        return self.execute(
            f"DELETE FROM {table} WHERE timestamp < datetime('now', '-{days} days')"
        )

    def purge_all_old_records(self, days: Optional[int] = None) -> dict[str, int]:
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
        tables: list[str] = [
            "health_snapshots", "sensor_readings", "detections",
            "species_sightings", "species_catalog", "pest_behavior",
            "pest_events", "weed_detections", "surveillance_events",
            "microclimate_readings", "microclimate_surveys",
            "navigation_log", "mission_events", "patrol_log",
            "occupancy_grid", "rtb_events", "system_events",
            "confidence_tracking"
        ]
        stats: dict[str, Any] = {}
        for table in tables:
            try:
                stats[table] = self.get_table_count(table)
            except DatabaseError:
                stats[table] = -1
        try:
            stats["file_size_mb"] = round(os.path.getsize(self._db_path) / (1024 * 1024), 2)
        except OSError:
            stats["file_size_mb"] = 0.0
        stats["total_rows"] = sum(v for v in stats.values() if isinstance(v, int) and v > 0)
        return stats

    def vacuum(self) -> None:
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
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
                self._local.connection = None
                logger.debug("Database connection closed for thread %s", threading.current_thread().name)
            except sqlite3.Error as e:
                logger.error("Error closing database connection: %s", e)

    def close_all(self) -> None:
        self.close()
        logger.info("Database shutdown complete")

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance.close()
        cls._instance = None
        cls._initialized = False

    def __repr__(self) -> str:
        return f"CerberusDB(path='{self._db_path}')"