"""
Cerberus Integration Smoke Test
Boots all subsystems in simulation mode (no hardware) and validates
they initialize, communicate, and shut down cleanly. This is the
final gate before deploying to the Pi.

What it proves:
    - Config loads and validates
    - Database initializes with full schema
    - Config validator runs clean on test config
    - Head detector identifies a head via config override
    - Obstacle detector starts/stops in simulation mode
    - Path planner creates grid, plans paths, persists to DB
    - Behavior tree builds, ticks, and reacts to context
    - Adaptive learner binds to DB and runs learning pass
    - Logger initializes with all handlers
    - All subsystems shut down without errors
"""

import os
import time
import logging
import pytest
from typing import Any

from cerberus.core.config import CerberusConfig
from cerberus.core.config_validator import ConfigValidator, ValidationResult, validate_config
from cerberus.storage.db import CerberusDB
from cerberus.heads.head_detector import (
    HeadDetector, HeadDetectionResult, HeadType, DetectionMethod
)
from cerberus.perception.obstacle import (
    ObstacleDetector, ObstacleMap, ObstacleZone, AvoidanceDirection
)
from cerberus.intelligence.path_planner import (
    PathPlanner, PlannedPath, CellState
)
from cerberus.intelligence.behavior_tree import (
    BehaviorTree, BehaviorContext, NodeStatus
)
from cerberus.intelligence.adaptive_learner import AdaptiveLearner


HOME_LAT: float = 36.1699
HOME_LON: float = -115.1398


class TestFullBootSequence:
    """
    Simulates the Cerberus boot sequence end-to-end.
    Each test method represents a phase of startup.
    """

    def test_phase_1_config_loads(self, config: CerberusConfig) -> None:
        """Config file parses and all sections accessible."""
        assert config.get("system", "name") == "cerberus-test"
        assert config.system is not None
        assert config.database is not None
        assert config.mqtt is not None
        assert config.safety is not None
        assert config.motors is not None
        assert config.navigation is not None
        assert config.heads is not None

    def test_phase_2_config_validates(self, config: CerberusConfig) -> None:
        """Config validator finds no errors on valid config."""
        result: ValidationResult = validate_config(config)
        assert result.valid, (
            f"Config validation failed:\n" +
            "\n".join(f"  {i.severity.value}: {i.path} — {i.message}" for i in result.issues)
        )

    def test_phase_3_database_initializes(self, db: CerberusDB) -> None:
        """Database creates all 18 tables with indexes."""
        stats: dict[str, Any] = db.get_database_stats()
        assert stats["health_snapshots"] == 0
        assert stats["system_events"] == 0

        tables: list[dict[str, Any]] = db.query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names: list[str] = [t["name"] for t in tables]
        assert len(table_names) >= 18

    def test_phase_4_logger_initializes(self, config: CerberusConfig, temp_dir: str) -> None:
        """Logging system sets up handlers and creates log files."""
        import cerberus.core.logger as log_module
        log_module._logger_initialized = False
        log_module._mqtt_handler = None
        logging.getLogger().handlers.clear()

        log_dir: str = os.path.join(temp_dir, "smoke_logs")
        config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(config)

        root: logging.Logger = logging.getLogger()
        assert len(root.handlers) >= 3

        logger: logging.Logger = logging.getLogger("cerberus.smoke")
        logger.info("Smoke test boot")
        logger.warning("Smoke test warning")
        logger.error("Smoke test error")

        assert os.path.exists(os.path.join(log_dir, "cerberus.log"))
        assert os.path.exists(os.path.join(log_dir, "cerberus_errors.log"))
        assert get_mqtt_log_handler() is not None

        log_module._logger_initialized = False
        log_module._mqtt_handler = None
        logging.getLogger().handlers.clear()

    def test_phase_5_head_detection(self, config: CerberusConfig) -> None:
        """Head detector identifies surveillance head via config override."""
        config._data["heads"]["active_head"] = "surveillance"
        detector: HeadDetector = HeadDetector(config)

        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.SURVEILLANCE
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        assert result.confidence == 1.0

        cls = detector.get_head_class(result.head_type)
        assert cls is not None

        detector.close()

    def test_phase_6_obstacle_detector(self, config: CerberusConfig) -> None:
        """Obstacle detector starts, polls, and stops in simulation."""
        config._data["obstacle"]["poll_interval"] = 0.05
        detector: ObstacleDetector = ObstacleDetector(config)

        assert detector.hardware_available is False

        started: bool = detector.start()
        assert started is True
        assert detector.is_running is True

        time.sleep(0.3)

        obs_map: ObstacleMap = detector.get_obstacle_map()
        assert obs_map.timestamp > 0
        assert obs_map.front.valid is True

        rec = detector.get_avoidance_recommendation()
        assert rec.zone == ObstacleZone.CLEAR
        assert rec.direction == AvoidanceDirection.NONE

        detector.stop()
        assert detector.is_running is False
        assert detector.total_readings > 0

    def test_phase_7_path_planner(self, config: CerberusConfig, db: CerberusDB) -> None:
        """Path planner builds grid, plans paths, and persists obstacles."""
        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)

        planner.update_from_position(HOME_LAT, HOME_LON)
        assert planner.grid.free_count >= 1

        planner.update_from_obstacle(
            rover_lat=HOME_LAT, rover_lon=HOME_LON,
            distance_cm=200.0, heading_deg=0.0
        )
        assert planner.grid.obstacle_count >= 1

        goal_lat: float = HOME_LAT + 0.0005
        result: PlannedPath = planner.plan_path(HOME_LAT, HOME_LON, goal_lat, HOME_LON)
        assert result.success is True
        assert len(result.waypoints) >= 2
        assert result.distance_m > 0
        assert result.planning_time_ms >= 0

        saved: int = planner.save_grid_to_db()
        assert saved > 0

        grid_rows: list[dict[str, Any]] = db.get_occupancy_grid()
        assert len(grid_rows) > 0

        s: dict[str, Any] = planner.stats()
        assert s["total_plans"] == 1
        assert s["successful_plans"] == 1
        assert s["success_rate"] == 1.0

    def test_phase_8_behavior_tree(self, config: CerberusConfig) -> None:
        """Behavior tree builds, ticks, and reacts to context changes."""
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        transitions: list[tuple[str, str]] = []
        tree.set_behavior_change_callback(lambda old, new: transitions.append((old, new)))

        ctx: BehaviorContext = BehaviorContext(
            gps_fix=True, at_home=True, distance_to_home_m=0.0
        )
        status: NodeStatus = tree.tick(ctx)
        assert tree.active_behavior == "idle_patrol"
        assert tree.tick_count == 1

        ctx.mission_active = True
        ctx.mission_paused = False
        tree.tick(ctx)
        assert tree.active_behavior == "mission_execution"

        ctx.obstacle_path_clear = False
        ctx.obstacle_zone = "caution"
        ctx.obstacle_closest_cm = 40.0
        tree.tick(ctx)
        assert tree.active_behavior == "obstacle_avoidance"

        ctx.safety_violation = True
        ctx.safety_reason = "smoke test emergency"
        tree.tick(ctx)
        assert tree.active_behavior == "emergency_stop"
        assert ctx.requesting_stop is True

        assert len(transitions) >= 3

        structure: list[dict[str, Any]] = tree.get_tree_structure()
        assert len(structure) > 0
        assert structure[0]["name"] == "CerberusRoot"

    def test_phase_9_adaptive_learner(self, config: CerberusConfig, db: CerberusDB) -> None:
        """Adaptive learner binds to DB and runs all learning modules."""
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        results: dict[str, Any] = learner.run_all()
        assert results["run_number"] == 1
        assert "duration_ms" in results
        assert learner.total_runs == 1

        s: dict[str, Any] = learner.stats()
        assert s["total_runs"] == 1
        assert s["running"] is False

    def test_phase_10_database_logging(self, db: CerberusDB) -> None:
        """All database convenience methods work for a simulated mission."""
        db.log_system_event(
            event_type="boot", severity="INFO",
            source="smoke_test", message="Smoke test boot"
        )

        db.log_health(
            cpu_pct=35.0, memory_pct=45.0, disk_pct=20.0,
            cpu_temp_c=52.0, battery_voltage=12.1, battery_current_a=1.5,
            battery_pct=85.0, gps_lat=HOME_LAT, gps_lon=HOME_LON
        )

        db.log_sensor_reading(
            sensor_type="bme680", temperature_c=38.5,
            humidity_pct=22.0, pressure_hpa=1013.25
        )

        db.log_detection(
            head_type="surveillance", detection_type="motion",
            label="cat", confidence=0.78,
            gps_lat=HOME_LAT, gps_lon=HOME_LON,
            inference_time_ms=32.5
        )

        db.log_surveillance_event(
            event_type="motion", threat_level="low",
            label="cat", confidence=0.78,
            motion_pct=4.2, gps_lat=HOME_LAT, gps_lon=HOME_LON
        )

        db.log_navigation_event(
            event_type="waypoint_reached",
            target_lat=HOME_LAT + 0.0001, target_lon=HOME_LON,
            actual_lat=HOME_LAT + 0.00009, actual_lon=HOME_LON + 0.00001,
            distance_m=1.1, heading_deg=0.0,
            waypoint_name="smoke_wp_1", status="success"
        )

        db.log_mission_event(
            mission_id="SMOKE001", mission_name="smoke_patrol",
            event_type="started", head_type="surveillance",
            message="Smoke test mission"
        )

        db.log_patrol_event(
            route_name="smoke_route", event_type="started",
            loop_number=1, gps_lat=HOME_LAT, gps_lon=HOME_LON
        )

        db.log_confidence(
            model_name="smoke_model", label="test",
            predicted_confidence=0.75
        )

        db.log_system_event(
            event_type="shutdown", severity="INFO",
            source="smoke_test", message="Smoke test shutdown"
        )

        stats: dict[str, Any] = db.get_database_stats()
        assert stats["health_snapshots"] >= 1
        assert stats["sensor_readings"] >= 1
        assert stats["detections"] >= 1
        assert stats["system_events"] >= 2
        assert stats["navigation_log"] >= 1
        assert stats["mission_events"] >= 1
        assert stats["patrol_log"] >= 1
        assert stats["surveillance_events"] >= 1
        assert stats["confidence_tracking"] >= 1


class TestCleanShutdown:
    """All subsystems release resources cleanly."""

    def test_shutdown_sequence(self, config: CerberusConfig, db: CerberusDB) -> None:
        """Boot everything, then shut it all down without errors."""
        config._data["obstacle"]["poll_interval"] = 0.05
        config._data["heads"]["active_head"] = "surveillance"

        detector: HeadDetector = HeadDetector(config)
        detector.detect()

        obstacle: ObstacleDetector = ObstacleDetector(config)
        obstacle.start()

        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)

        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        time.sleep(0.2)

        ctx: BehaviorContext = BehaviorContext(gps_fix=True, at_home=True)
        tree.tick(ctx)

        learner.stop()
        assert learner.is_running is False

        obstacle.stop()
        assert obstacle.is_running is False

        planner.save_grid_to_db()

        detector.close()

        db.log_system_event(
            event_type="shutdown", severity="INFO",
            source="smoke_test", message="Clean shutdown complete"
        )

        events: list[dict[str, Any]] = db.get_system_events(hours=1)
        shutdown_events: list[dict[str, Any]] = [
            e for e in events if e["event_type"] == "shutdown"
        ]
        assert len(shutdown_events) >= 1


class TestCrossSubsystemFlow:
    """Data flows between subsystems correctly."""

    def test_obstacle_feeds_path_planner(self, config: CerberusConfig, db: CerberusDB) -> None:
        """Obstacle detection writes to grid, planner routes around it."""
        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)

        planner.update_from_position(HOME_LAT, HOME_LON)

        planner.update_from_obstacle(
            rover_lat=HOME_LAT, rover_lon=HOME_LON,
            distance_cm=100.0, heading_deg=0.0
        )

        path_through: PlannedPath = planner.plan_path(
            HOME_LAT, HOME_LON,
            HOME_LAT + 0.001, HOME_LON
        )
        assert path_through.success is True

        obstacles: list = planner.grid.get_obstacles()
        assert len(obstacles) >= 1

    def test_detection_feeds_behavior_tree(self, config: CerberusConfig) -> None:
        """AI detection triggers investigation behavior."""
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        ctx: BehaviorContext = BehaviorContext(
            gps_fix=True,
            active_detection=True,
            threat_detected=False,
            detection_type="weed",
            detection_label="dandelion",
            detection_confidence=0.85,
            gps_lat=HOME_LAT,
            gps_lon=HOME_LON
        )
        tree.tick(ctx)
        assert tree.active_behavior == "investigation"
        assert ctx.requesting_investigation is True
        assert ctx.blackboard.get("investigation_label") == "dandelion"

    def test_safety_overrides_mission(self, config: CerberusConfig) -> None:
        """Safety violation preempts active mission."""
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        ctx: BehaviorContext = BehaviorContext(
            mission_active=True, gps_fix=True
        )
        tree.tick(ctx)
        assert tree.active_behavior == "mission_execution"

        ctx.battery_pct = 5.0
        ctx.distance_to_home_m = 50.0
        tree.tick(ctx)
        assert tree.active_behavior == "return_to_base"
        assert ctx.requesting_rtb is True

    def test_learner_uses_db_data(self, config: CerberusConfig, db: CerberusDB) -> None:
        """Learner reads from DB tables populated by other subsystems."""
        for i in range(15):
            db.log_sensor_reading(
                sensor_type="bme680",
                temperature_c=38.0 + (i % 5) * 0.5,
                humidity_pct=20.0 + i
            )

        for i in range(10):
            db.log_detection(
                head_type="surveillance", detection_type="motion",
                label="event", confidence=0.7,
                gps_lat=HOME_LAT + 0.00001 * i,
                gps_lon=HOME_LON
            )

        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        results: dict[str, Any] = learner.run_all()
        assert results["run_number"] == 1
        assert "duration_ms" in results