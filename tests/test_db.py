"""
Tests for cerberus.storage.db — CerberusDB
Validates schema creation, all 18 tables, convenience methods,
JSON handling, singleton behavior, transactions, and purge operations.
"""

import time
import json
import pytest
from typing import Any

from cerberus.storage.db import CerberusDB


class TestDBInitialization:
    """Database creation and schema validation."""

    def test_db_created(self, db: CerberusDB) -> None:
        assert db is not None

    def test_all_tables_exist(self, db: CerberusDB) -> None:
        expected: list[str] = [
            "health_snapshots", "sensor_readings", "detections",
            "species_sightings", "species_catalog", "pest_behavior",
            "pest_events", "weed_detections", "surveillance_events",
            "microclimate_readings", "microclimate_surveys",
            "navigation_log", "mission_events", "patrol_log",
            "occupancy_grid", "rtb_events", "system_events",
            "confidence_tracking"
        ]
        rows: list[dict[str, Any]] = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables: list[str] = [r["name"] for r in rows]
        for table in expected:
            assert table in tables, f"Missing table: {table}"

    def test_wal_mode_enabled(self, db: CerberusDB) -> None:
        rows: list[dict[str, Any]] = db.query("PRAGMA journal_mode")
        mode: str = rows[0].get("journal_mode", "")
        assert mode.lower() == "wal"

    def test_singleton_behavior(self, db: CerberusDB) -> None:
        second: CerberusDB = CerberusDB()
        assert db is second
        CerberusDB.reset()


class TestHealthSnapshots:
    """Health logging and retrieval."""

    def test_log_health(self, db: CerberusDB) -> None:
        db.log_health(
            cpu_pct=45.0, memory_pct=60.0, disk_pct=30.0,
            cpu_temp_c=55.0, battery_voltage=11.8, battery_current_a=2.1,
            battery_pct=72.0, gps_lat=36.17, gps_lon=-115.14
        )
        result: dict[str, Any] = db.get_latest_health()
        assert result is not None
        assert result["cpu_pct"] == 45.0
        assert result["battery_voltage"] == 11.8

    def test_get_health_history(self, db: CerberusDB) -> None:
        for i in range(5):
            db.log_health(cpu_pct=float(i * 10), memory_pct=50.0, disk_pct=30.0)
        history: list[dict[str, Any]] = db.get_health_history(hours=1)
        assert len(history) == 5

    def test_get_latest_health_empty(self, db: CerberusDB) -> None:
        result = db.get_latest_health()
        assert result is None


class TestSensorReadings:
    """Sensor data logging."""

    def test_log_sensor_reading(self, db: CerberusDB) -> None:
        db.log_sensor_reading(
            sensor_type="bme680",
            temperature_c=38.5,
            humidity_pct=22.0,
            pressure_hpa=1013.25
        )
        readings: list[dict[str, Any]] = db.get_sensor_readings("bme680", hours=1)
        assert len(readings) == 1
        assert readings[0]["temperature_c"] == 38.5

    def test_multiple_sensor_types(self, db: CerberusDB) -> None:
        db.log_sensor_reading(sensor_type="bme680", temperature_c=38.0)
        db.log_sensor_reading(sensor_type="scd40", co2_ppm=420.0)
        db.log_sensor_reading(sensor_type="sht45", temperature_c=37.0, humidity_pct=25.0)

        bme: list[dict[str, Any]] = db.get_sensor_readings("bme680", hours=1)
        scd: list[dict[str, Any]] = db.get_sensor_readings("scd40", hours=1)
        sht: list[dict[str, Any]] = db.get_sensor_readings("sht45", hours=1)

        assert len(bme) == 1
        assert len(scd) == 1
        assert len(sht) == 1


class TestDetections:
    """AI detection logging and querying."""

    def test_log_detection(self, db: CerberusDB) -> None:
        db.log_detection(
            head_type="weed_scanner",
            detection_type="weed",
            label="dandelion",
            confidence=0.87,
            gps_lat=36.17,
            gps_lon=-115.14,
            inference_time_ms=45.2
        )
        results: list[dict[str, Any]] = db.get_detections_by_type("weed", hours=1)
        assert len(results) == 1
        assert results[0]["label"] == "dandelion"
        assert results[0]["confidence"] == 0.87

    def test_get_detections_since(self, db: CerberusDB) -> None:
        for i in range(3):
            db.log_detection(
                head_type="surveillance",
                detection_type="motion",
                label=f"event_{i}",
                confidence=0.5 + i * 0.1
            )
        results: list[dict[str, Any]] = db.get_detections_since(hours=1)
        assert len(results) == 3

    def test_detection_stats(self, db: CerberusDB) -> None:
        db.log_detection(head_type="surveillance", detection_type="threat", label="person", confidence=0.9)
        db.log_detection(head_type="surveillance", detection_type="threat", label="person", confidence=0.8)
        db.log_detection(head_type="surveillance", detection_type="threat", label="animal", confidence=0.7)

        stats: list[dict[str, Any]] = db.get_detection_stats(hours=1)
        assert len(stats) >= 1


class TestSpecies:
    """Species sighting and catalog tracking."""

    def test_log_sighting(self, db: CerberusDB) -> None:
        db.log_sighting(
            species="Northern Mockingbird",
            common_name="Northern Mockingbird",
            scientific_name="Mimus polyglottos",
            identification_method="visual",
            confidence=0.92,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        sightings: list[dict[str, Any]] = db.get_recent_sightings(limit=10)
        assert len(sightings) == 1
        assert sightings[0]["species"] == "Northern Mockingbird"

    def test_species_catalog_updated(self, db: CerberusDB) -> None:
        db.log_sighting(
            species="Gambel's Quail",
            common_name="Gambel's Quail",
            scientific_name="Callipepla gambelii",
            identification_method="visual",
            confidence=0.85
        )
        db.log_sighting(
            species="Gambel's Quail",
            common_name="Gambel's Quail",
            scientific_name="Callipepla gambelii",
            identification_method="audio",
            confidence=0.95
        )
        catalog: list[dict[str, Any]] = db.get_species_catalog()
        quail: list[dict[str, Any]] = [c for c in catalog if c["species"] == "Gambel's Quail"]
        assert len(quail) == 1
        assert quail[0]["sighting_count"] == 2
        assert quail[0]["best_confidence"] == 0.95


class TestPestBehavior:
    """Pest adaptive learning persistence."""

    def test_log_pest_event(self, db: CerberusDB) -> None:
        db.log_pest_event(
            species="rabbit",
            confidence=0.8,
            deterrent_action="audio_predator",
            escalation="gentle",
            response_effective=True,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        events: list[dict[str, Any]] = db.get_pest_activity(hours=1)
        assert len(events) == 1

    def test_update_pest_behavior(self, db: CerberusDB) -> None:
        db.update_pest_behavior(
            species="rabbit",
            encounters=5,
            deterred_count=3,
            ignored_count=2,
            effective_actions=["audio_predator", "led_flash"],
            current_escalation="moderate"
        )
        behavior: dict[str, Any] = db.get_pest_behavior("rabbit")
        assert behavior is not None
        assert behavior["encounters"] == 5
        assert behavior["deterred_count"] == 3

    def test_get_all_pest_behaviors(self, db: CerberusDB) -> None:
        db.update_pest_behavior(species="rabbit", encounters=1)
        db.update_pest_behavior(species="squirrel", encounters=2)
        behaviors: list[dict[str, Any]] = db.get_all_pest_behaviors()
        assert len(behaviors) >= 2


class TestWeedDetections:
    """Weed detection geotag logging."""

    def test_log_weed_detection(self, db: CerberusDB) -> None:
        db.log_weed_detection(
            species="dandelion",
            confidence=0.88,
            gps_lat=36.17,
            gps_lon=-115.14,
            hdop=1.2,
            scan_pan=45.0,
            scan_tilt=-10.0
        )
        unverified: list[dict[str, Any]] = db.get_unverified_weeds()
        assert len(unverified) == 1
        assert unverified[0]["species"] == "dandelion"

    def test_weed_hotspots(self, db: CerberusDB) -> None:
        for _ in range(3):
            db.log_weed_detection(
                species="crabgrass",
                confidence=0.75,
                gps_lat=36.17001,
                gps_lon=-115.13981
            )
        hotspots: list[dict[str, Any]] = db.get_weed_hotspots(min_detections=2)
        assert len(hotspots) >= 1


class TestSurveillance:
    """Surveillance event logging."""

    def test_log_surveillance_event(self, db: CerberusDB) -> None:
        db.log_surveillance_event(
            event_type="motion",
            threat_level="low",
            label="cat",
            confidence=0.72,
            region_count=1,
            motion_pct=5.3,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        threats: list[dict[str, Any]] = db.get_threat_history(hours=1)
        assert len(threats) == 1

    def test_motion_activity(self, db: CerberusDB) -> None:
        db.log_surveillance_event(event_type="motion", motion_pct=3.0)
        db.log_surveillance_event(event_type="motion", motion_pct=8.0)
        activity: list[dict[str, Any]] = db.get_motion_activity(hours=1)
        assert len(activity) == 2


class TestMicroclimate:
    """Microclimate data logging and survey tracking."""

    def test_log_microclimate_reading(self, db: CerberusDB) -> None:
        db.log_microclimate_reading(
            survey_name="test_survey",
            temperature_c=42.5,
            humidity_pct=18.0,
            gps_lat=36.17,
            gps_lon=-115.14,
            grid_row=0,
            grid_col=0
        )
        readings: list[dict[str, Any]] = db.get_survey_readings("test_survey")
        assert len(readings) == 1
        assert readings[0]["temperature_c"] == 42.5

    def test_log_microclimate_survey(self, db: CerberusDB) -> None:
        db.log_microclimate_survey(
            survey_name="test_survey",
            total_readings=25,
            duration_seconds=300.0,
            grid_rows=5,
            grid_cols=5,
            temp_min_c=38.0,
            temp_max_c=48.0,
            temp_avg_c=43.0,
            temp_stdev_c=2.5,
            humidity_min_pct=12.0,
            humidity_max_pct=25.0,
            humidity_avg_pct=18.0,
            hotspot_count=3,
            coldspot_count=1,
            moisture_zone_count=0
        )
        surveys: list[dict[str, Any]] = db.get_survey_list()
        assert len(surveys) == 1
        assert surveys[0]["survey_name"] == "test_survey"


class TestNavigation:
    """Navigation event logging."""

    def test_log_navigation_event(self, db: CerberusDB) -> None:
        db.log_navigation_event(
            event_type="waypoint_reached",
            target_lat=36.171,
            target_lon=-115.140,
            actual_lat=36.1709,
            actual_lon=-115.1401,
            distance_m=1.2,
            heading_deg=45.0,
            waypoint_name="wp_01",
            status="success"
        )
        history: list[dict[str, Any]] = db.get_navigation_history(hours=1)
        assert len(history) == 1
        assert history[0]["event_type"] == "waypoint_reached"


class TestMissionEvents:
    """Mission lifecycle logging."""

    def test_log_mission_event(self, db: CerberusDB) -> None:
        db.log_mission_event(
            mission_id="M001",
            mission_name="patrol_front_yard",
            event_type="started",
            head_type="surveillance",
            message="Mission started"
        )
        history: list[dict[str, Any]] = db.get_mission_history(hours=1)
        assert len(history) == 1
        assert history[0]["mission_id"] == "M001"


class TestPatrolLog:
    """Patrol route logging."""

    def test_log_patrol_event(self, db: CerberusDB) -> None:
        db.log_patrol_event(
            route_name="perimeter",
            event_type="waypoint_reached",
            waypoint_name="corner_ne",
            waypoint_index=2,
            loop_number=1,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        history: list[dict[str, Any]] = db.get_patrol_history(route_name="perimeter", hours=1)
        assert len(history) == 1


class TestOccupancyGrid:
    """Occupancy grid persistence."""

    def test_update_and_get_grid(self, db: CerberusDB) -> None:
        db.update_grid_cell(
            grid_x=5, grid_y=10,
            state="occupied",
            confidence=0.95,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        grid: list[dict[str, Any]] = db.get_occupancy_grid()
        assert len(grid) >= 1
        cell: dict[str, Any] = grid[0]
        assert cell["grid_x"] == 5
        assert cell["grid_y"] == 10
        assert cell["state"] == "occupied"

    def test_get_obstacles(self, db: CerberusDB) -> None:
        db.update_grid_cell(grid_x=1, grid_y=1, state="occupied")
        db.update_grid_cell(grid_x=2, grid_y=2, state="free")
        db.update_grid_cell(grid_x=3, grid_y=3, state="occupied")

        obstacles: list[dict[str, Any]] = db.get_obstacles()
        assert len(obstacles) == 2

    def test_unique_grid_cell(self, db: CerberusDB) -> None:
        db.update_grid_cell(grid_x=5, grid_y=5, state="free", confidence=0.5)
        db.update_grid_cell(grid_x=5, grid_y=5, state="occupied", confidence=1.0)
        grid: list[dict[str, Any]] = db.get_occupancy_grid()
        matching: list[dict[str, Any]] = [c for c in grid if c["grid_x"] == 5 and c["grid_y"] == 5]
        assert len(matching) == 1
        assert matching[0]["state"] == "occupied"


class TestRTBEvents:
    """Return-to-base logging."""

    def test_log_rtb_event(self, db: CerberusDB) -> None:
        db.log_rtb_event(
            reason="battery_critical",
            trigger_value=12.5,
            start_lat=36.171,
            start_lon=-115.140,
            home_lat=36.170,
            home_lon=-115.140,
            distance_m=11.1,
            result="success",
            duration_seconds=45.0
        )
        history: list[dict[str, Any]] = db.get_rtb_history(hours=1)
        assert len(history) == 1
        assert history[0]["reason"] == "battery_critical"


class TestSystemEvents:
    """System event logging."""

    def test_log_system_event(self, db: CerberusDB) -> None:
        db.log_system_event(
            event_type="boot",
            severity="INFO",
            source="brain",
            message="Cerberus booting up"
        )
        events: list[dict[str, Any]] = db.get_system_events(hours=1)
        assert len(events) == 1
        assert events[0]["event_type"] == "boot"

    def test_system_event_with_metadata(self, db: CerberusDB) -> None:
        meta: dict[str, Any] = {"version": "1.0.0", "uptime": 0}
        db.log_system_event(
            event_type="boot",
            severity="INFO",
            source="brain",
            message="Boot with metadata",
            metadata=meta
        )
        events: list[dict[str, Any]] = db.get_system_events(hours=1)
        assert len(events) == 1
        parsed: Any = events[0].get("metadata")
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        assert parsed["version"] == "1.0.0"


class TestConfidenceTracking:
    """Model confidence calibration tracking."""

    def test_log_and_verify_prediction(self, db: CerberusDB) -> None:
        db.log_confidence(
            model_name="weed_detector",
            label="dandelion",
            predicted_confidence=0.85
        )

        rows: list[dict[str, Any]] = db.query(
            "SELECT id FROM confidence_tracking ORDER BY id DESC LIMIT 1"
        )
        assert len(rows) == 1
        row_id: int = rows[0]["id"]

        db.verify_prediction(row_id, correct=True, notes="Confirmed dandelion")

        verified: list[dict[str, Any]] = db.query(
            "SELECT * FROM confidence_tracking WHERE id = ?", (row_id,)
        )
        assert verified[0]["verified"] == 1
        assert verified[0]["correct"] == 1

    def test_model_accuracy(self, db: CerberusDB) -> None:
        for i in range(10):
            db.log_confidence(
                model_name="test_model",
                label="weed",
                predicted_confidence=0.7 + i * 0.02
            )
        rows: list[dict[str, Any]] = db.query(
            "SELECT id FROM confidence_tracking WHERE model_name = 'test_model'"
        )
        for i, row in enumerate(rows):
            db.verify_prediction(row["id"], correct=(i < 8))

        accuracy: dict[str, Any] = db.get_model_accuracy("test_model")
        assert accuracy is not None
        assert accuracy["total_verified"] == 10
        assert accuracy["correct_count"] == 8

    def test_false_positive_rate(self, db: CerberusDB) -> None:
        for i in range(5):
            db.log_confidence(model_name="fp_model", label="threat", predicted_confidence=0.6)
        rows: list[dict[str, Any]] = db.query(
            "SELECT id FROM confidence_tracking WHERE model_name = 'fp_model'"
        )
        for i, row in enumerate(rows):
            db.verify_prediction(row["id"], correct=(i < 2))

        fp_rate: float = db.get_false_positive_rate("fp_model")
        assert 0.5 < fp_rate < 0.7


class TestDatabaseMaintenance:
    """Purge and vacuum operations."""

    def test_get_database_stats(self, db: CerberusDB) -> None:
        db.log_health(cpu_pct=50.0)
        db.log_system_event(event_type="test", severity="INFO", source="test", message="test")
        stats: dict[str, Any] = db.get_database_stats()
        assert stats["health_snapshots"] >= 1
        assert stats["system_events"] >= 1

    def test_vacuum(self, db: CerberusDB) -> None:
        db.log_health(cpu_pct=50.0)
        db.vacuum()


class TestTransactions:
    """Transaction context manager."""

    def test_transaction_commit(self, db: CerberusDB) -> None:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO system_events (event_type, severity, source, message) "
                "VALUES (?, ?, ?, ?)",
                ("test", "INFO", "test", "transaction test")
            )
        events: list[dict[str, Any]] = db.get_system_events(hours=1)
        assert any(e["message"] == "transaction test" for e in events)

    def test_transaction_rollback(self, db: CerberusDB) -> None:
        try:
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO system_events (event_type, severity, source, message) "
                    "VALUES (?, ?, ?, ?)",
                    ("rollback_test", "INFO", "test", "should not persist")
                )
                raise ValueError("Force rollback")
        except ValueError:
            pass

        events: list[dict[str, Any]] = db.get_system_events(hours=1)
        assert not any(e["message"] == "should not persist" for e in events)