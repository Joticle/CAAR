#!/usr/bin/env python3
"""
Cerberus Standalone Smoke Test
Run directly on the Pi or dev machine to validate all subsystems boot,
communicate, and shut down cleanly without requiring pytest.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --config config/cerberus.yaml
    python scripts/smoke_test.py --verbose

Exit codes:
    0 = All tests passed
    1 = One or more tests failed
    2 = Fatal error during setup
"""

import os
import sys
import time
import argparse
import tempfile
import traceback
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class TestResult:
    name: str = ""
    passed: bool = False
    duration_ms: float = 0.0
    message: str = ""
    error: str = ""


@dataclass
class SmokeResults:
    results: list[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


class SmokeTester:
    """Runs all smoke tests and collects results."""

    def __init__(self, config_path: Optional[str] = None, verbose: bool = False) -> None:
        self._config_path: Optional[str] = config_path
        self._verbose: bool = verbose
        self._results: SmokeResults = SmokeResults()
        self._config: Any = None
        self._db: Any = None
        self._temp_dir: str = ""

    def run_all(self) -> SmokeResults:
        self._results.start_time = time.time()
        self._print_header()

        try:
            self._setup()
        except Exception as e:
            self._print_fatal(f"Setup failed: {e}")
            self._results.end_time = time.time()
            return self._results

        tests: list[tuple[str, Callable]] = [
            ("Config Load", self._test_config_load),
            ("Config Validate", self._test_config_validate),
            ("Database Init", self._test_database_init),
            ("Database Schema", self._test_database_schema),
            ("Database CRUD", self._test_database_crud),
            ("Logger Init", self._test_logger_init),
            ("Head Detection", self._test_head_detection),
            ("Obstacle Detector", self._test_obstacle_detector),
            ("Path Planner", self._test_path_planner),
            ("Behavior Tree", self._test_behavior_tree),
            ("Adaptive Learner", self._test_adaptive_learner),
            ("Cross-System Flow", self._test_cross_system),
            ("Clean Shutdown", self._test_clean_shutdown),
        ]

        for name, test_fn in tests:
            self._run_test(name, test_fn)

        self._teardown()
        self._results.end_time = time.time()
        self._print_summary()
        return self._results

    def _setup(self) -> None:
        self._temp_dir = tempfile.mkdtemp(prefix="cerberus_smoke_")
        self._print_info(f"Temp directory: {self._temp_dir}")

        if self._config_path and os.path.exists(self._config_path):
            self._print_info(f"Using config: {self._config_path}")
        else:
            self._config_path = self._create_test_config()
            self._print_info(f"Using generated test config")

    def _teardown(self) -> None:
        if self._db:
            try:
                self._db.close()
                from cerberus.storage.db import CerberusDB
                CerberusDB._instance = None
                CerberusDB._initialized = False
            except Exception:
                pass

        if self._config:
            try:
                from cerberus.core.config import CerberusConfig
                CerberusConfig._instance = None
                CerberusConfig._initialized = False
            except Exception:
                pass

        import logging
        logging.getLogger().handlers.clear()

    def _run_test(self, name: str, test_fn: Callable) -> None:
        start: float = time.time()
        result: TestResult = TestResult(name=name)

        try:
            message: str = test_fn()
            result.passed = True
            result.message = message or "OK"
        except AssertionError as e:
            result.passed = False
            result.error = str(e)
        except Exception as e:
            result.passed = False
            result.error = f"{type(e).__name__}: {e}"
            if self._verbose:
                result.error += f"\n{traceback.format_exc()}"

        result.duration_ms = (time.time() - start) * 1000
        self._results.results.append(result)
        self._print_result(result)

    def _test_config_load(self) -> str:
        from cerberus.core.config import CerberusConfig
        CerberusConfig._instance = None
        CerberusConfig._initialized = False

        self._config = CerberusConfig(self._config_path)
        assert self._config.system is not None, "system section missing"
        assert self._config.database is not None, "database section missing"
        assert self._config.mqtt is not None, "mqtt section missing"
        assert self._config.safety is not None, "safety section missing"
        assert self._config.motors is not None, "motors section missing"

        name: str = self._config.get("system", "name", default="unknown")
        return f"Loaded: {name}"

    def _test_config_validate(self) -> str:
        from cerberus.core.config_validator import validate_config, ValidationResult
        result: ValidationResult = validate_config(self._config)

        if not result.valid:
            errors: str = "; ".join(
                f"{i.path}: {i.message}" for i in result.issues
                if i.severity.value == "error"
            )
            assert False, f"Validation errors: {errors}"

        return f"{result.error_count} errors, {result.warning_count} warnings"

    def _test_database_init(self) -> str:
        from cerberus.storage.db import CerberusDB
        CerberusDB._instance = None
        CerberusDB._initialized = False

        db_path: str = os.path.join(self._temp_dir, "smoke_test.db")
        self._config._data["database"]["path"] = db_path

        self._db = CerberusDB(self._config)
        assert os.path.exists(db_path), "Database file not created"

        size: int = os.path.getsize(db_path)
        return f"Created: {db_path} ({size:,} bytes)"

    def _test_database_schema(self) -> str:
        tables: list[dict[str, Any]] = self._db.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        table_names: list[str] = sorted([t["name"] for t in tables])

        required: list[str] = [
            "health_snapshots", "sensor_readings", "detections",
            "species_sightings", "species_catalog", "pest_behavior",
            "pest_events", "weed_detections", "surveillance_events",
            "microclimate_readings", "microclimate_surveys",
            "navigation_log", "mission_events", "patrol_log",
            "occupancy_grid", "rtb_events", "system_events",
            "confidence_tracking"
        ]

        missing: list[str] = [t for t in required if t not in table_names]
        assert len(missing) == 0, f"Missing tables: {missing}"

        indexes: list[dict[str, Any]] = self._db.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )

        return f"{len(table_names)} tables, {len(indexes)} indexes"

    def _test_database_crud(self) -> str:
        self._db.log_system_event(
            event_type="smoke_test", severity="INFO",
            source="smoke_tester", message="CRUD validation"
        )

        self._db.log_health(
            cpu_pct=30.0, memory_pct=40.0, disk_pct=15.0,
            cpu_temp_c=50.0, battery_voltage=12.0, battery_current_a=1.0,
            battery_pct=90.0, gps_lat=36.1699, gps_lon=-115.1398
        )

        self._db.log_sensor_reading(
            sensor_type="bme680", temperature_c=38.0,
            humidity_pct=20.0, pressure_hpa=1013.0
        )

        self._db.log_detection(
            head_type="surveillance", detection_type="motion",
            label="test", confidence=0.75,
            gps_lat=36.1699, gps_lon=-115.1398
        )

        health = self._db.get_latest_health()
        assert health is not None, "Health snapshot not retrieved"

        stats: dict[str, Any] = self._db.get_database_stats()
        assert stats["health_snapshots"] >= 1
        assert stats["sensor_readings"] >= 1
        assert stats["detections"] >= 1
        assert stats["system_events"] >= 1

        return f"Write/read verified across {len(stats)} tables"

    def _test_logger_init(self) -> str:
        import logging
        import cerberus.core.logger as log_module

        log_module._logger_initialized = False
        log_module._mqtt_handler = None
        logging.getLogger().handlers.clear()

        log_dir: str = os.path.join(self._temp_dir, "logs")
        self._config._data["system"]["log_dir"] = log_dir

        from cerberus.core.logger import setup_logging, get_mqtt_log_handler
        setup_logging(self._config)

        logger: logging.Logger = logging.getLogger("cerberus.smoke")
        logger.info("Smoke test info")
        logger.warning("Smoke test warning")
        logger.error("Smoke test error")

        assert os.path.exists(os.path.join(log_dir, "cerberus.log"))
        assert os.path.exists(os.path.join(log_dir, "cerberus_errors.log"))
        assert get_mqtt_log_handler() is not None

        handler_count: int = len(logging.getLogger().handlers)

        log_module._logger_initialized = False
        log_module._mqtt_handler = None
        logging.getLogger().handlers.clear()

        return f"{handler_count} handlers, log files created"

    def _test_head_detection(self) -> str:
        from cerberus.heads.head_detector import (
            HeadDetector, HeadDetectionResult, HeadType, DetectionMethod
        )

        self._config._data["heads"]["active_head"] = "surveillance"
        detector: HeadDetector = HeadDetector(self._config)

        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.SURVEILLANCE
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        assert result.confidence == 1.0

        cls = detector.get_head_class(result.head_type)
        assert cls is not None

        detector.close()
        return f"Detected: {result.head_type.value} via {result.method.value}"

    def _test_obstacle_detector(self) -> str:
        from cerberus.perception.obstacle import (
            ObstacleDetector, ObstacleZone
        )

        self._config._data["obstacle"]["poll_interval"] = 0.05
        detector: ObstacleDetector = ObstacleDetector(self._config)

        assert detector.hardware_available is False

        started: bool = detector.start()
        assert started, "Failed to start obstacle detector"

        time.sleep(0.3)

        obs_map = detector.get_obstacle_map()
        assert obs_map.timestamp > 0
        assert obs_map.front.valid

        rec = detector.get_avoidance_recommendation()
        readings: int = detector.total_readings

        detector.stop()
        assert not detector.is_running

        return f"Simulated {readings} readings, zone: {rec.zone.value}"

    def _test_path_planner(self) -> str:
        from cerberus.intelligence.path_planner import PathPlanner, PlannedPath

        planner: PathPlanner = PathPlanner(self._config)
        planner.bind_db(self._db)

        planner.update_from_position(36.1699, -115.1398)
        planner.update_from_obstacle(
            rover_lat=36.1699, rover_lon=-115.1398,
            distance_cm=200.0, heading_deg=0.0
        )

        result: PlannedPath = planner.plan_path(
            36.1699, -115.1398,
            36.1704, -115.1398
        )
        assert result.success, f"Path planning failed: {result.message}"

        saved: int = planner.save_grid_to_db()

        return (
            f"Path: {len(result.waypoints)} waypoints, "
            f"{result.distance_m:.1f}m, "
            f"{result.planning_time_ms:.1f}ms, "
            f"{saved} cells saved"
        )

    def _test_behavior_tree(self) -> str:
        from cerberus.intelligence.behavior_tree import (
            BehaviorTree, BehaviorContext, NodeStatus
        )

        tree: BehaviorTree = BehaviorTree(self._config)
        tree.build_default_tree()

        transitions: list[str] = []
        tree.set_behavior_change_callback(lambda o, n: transitions.append(n))

        ctx: BehaviorContext = BehaviorContext(
            gps_fix=True, at_home=True, distance_to_home_m=0.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "idle_patrol"

        ctx.mission_active = True
        ctx.mission_paused = False
        tree.tick(ctx)
        assert tree.active_behavior == "mission_execution"

        ctx.safety_violation = True
        ctx.safety_reason = "smoke_test"
        tree.tick(ctx)
        assert tree.active_behavior == "emergency_stop"
        assert ctx.requesting_stop

        structure = tree.get_tree_structure()
        node_count: int = len(structure)

        return f"{tree.tick_count} ticks, {len(transitions)} transitions, {node_count} nodes"

    def _test_adaptive_learner(self) -> str:
        from cerberus.intelligence.adaptive_learner import AdaptiveLearner

        learner: AdaptiveLearner = AdaptiveLearner(self._config)
        learner.bind_db(self._db)

        results: dict[str, Any] = learner.run_all()
        assert results["run_number"] == 1

        s: dict[str, Any] = learner.stats()
        assert s["total_runs"] == 1

        return f"Run #{results['run_number']} in {results['duration_ms']:.1f}ms"

    def _test_cross_system(self) -> str:
        from cerberus.intelligence.path_planner import PathPlanner, PlannedPath
        from cerberus.intelligence.behavior_tree import BehaviorTree, BehaviorContext

        planner: PathPlanner = PathPlanner.__new__(PathPlanner)
        planner._grid = __import__(
            "cerberus.intelligence.path_planner", fromlist=["OccupancyGrid"]
        ).OccupancyGrid(cell_size_m=0.5, home_lat=36.1699, home_lon=-115.1398)
        planner._db = self._db
        planner._unknown_penalty = 1.5
        planner._max_iterations = 10000
        planner._simplify_paths = True
        planner._simplify_tolerance_m = 1.0
        planner._total_plans = 0
        planner._successful_plans = 0
        planner._failed_plans = 0

        planner.update_from_position(36.1699, -115.1398)
        planner.update_from_obstacle(
            rover_lat=36.1699, rover_lon=-115.1398,
            distance_cm=150.0, heading_deg=45.0
        )

        result: PlannedPath = planner.plan_path(
            36.1699, -115.1398, 36.1702, -115.1395
        )
        assert result.success

        tree: BehaviorTree = BehaviorTree(self._config)
        tree.build_default_tree()

        ctx: BehaviorContext = BehaviorContext(
            gps_fix=True, active_detection=True, threat_detected=False,
            detection_type="weed", detection_label="dandelion",
            detection_confidence=0.85, gps_lat=36.1699, gps_lon=-115.1398
        )
        tree.tick(ctx)
        assert tree.active_behavior == "investigation"
        assert ctx.requesting_investigation

        return "Obstacle→planner and detection→behavior verified"

    def _test_clean_shutdown(self) -> str:
        self._db.log_system_event(
            event_type="shutdown", severity="INFO",
            source="smoke_tester", message="Clean shutdown"
        )

        self._db.vacuum()

        stats: dict[str, Any] = self._db.get_database_stats()
        total_rows: int = sum(v for v in stats.values() if isinstance(v, int))

        return f"Shutdown logged, DB vacuumed, {total_rows} total rows"

    def _create_test_config(self) -> str:
        import yaml

        config: dict[str, Any] = {
            "system": {
                "name": "cerberus-smoke",
                "version": "1.0.0",
                "environment": "development",
                "log_level": "DEBUG",
                "log_dir": os.path.join(self._temp_dir, "logs"),
                "project_root": self._temp_dir,
            },
            "database": {
                "path": os.path.join(self._temp_dir, "smoke.db"),
                "busy_timeout_ms": 3000,
                "wal_mode": True,
            },
            "i2c": {"bus": 1},
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "client_id": "cerberus-smoke",
                "qos": 1,
                "keepalive": 60,
                "topics": {
                    "health": "cerberus/telemetry/health",
                    "sensors": "cerberus/telemetry/sensors",
                    "detections": "cerberus/detections",
                    "mission": "cerberus/mission/status",
                    "commands": "cerberus/command",
                    "alerts": "cerberus/alerts",
                },
            },
            "safety": {
                "enabled": True,
                "battery_warn_pct": 30,
                "battery_critical_pct": 15,
                "battery_shutdown_pct": 5,
                "thermal_warn_c": 70,
                "thermal_critical_c": 80,
                "thermal_shutdown_c": 85,
                "watchdog_interval_s": 5,
                "max_mission_minutes": 120,
            },
            "power": {
                "monitor_type": "ina3221",
                "i2c_address": "0x40",
                "battery_cells": 3,
                "cell_min_v": 3.0,
                "cell_max_v": 4.2,
                "current_limit_a": 10.0,
            },
            "motors": {
                "left": {"pwm_pin": 12, "fwd_pin": 5, "rev_pin": 6},
                "right": {"pwm_pin": 13, "fwd_pin": 16, "rev_pin": 26},
                "frequency_hz": 1000,
                "ramp_steps": 10,
                "ramp_delay_ms": 20,
            },
            "servos": {
                "pca9685_address": "0x40",
                "pan_channel": 0,
                "tilt_channel": 1,
                "pan_range": [0, 180],
                "tilt_range": [0, 90],
            },
            "navigation": {
                "home_lat": 36.1699,
                "home_lon": -115.1398,
                "gps_port": "/dev/ttyAMA0",
                "gps_baud": 9600,
                "waypoint_radius_m": 2.0,
                "max_speed_mps": 1.5,
            },
            "camera": {
                "resolution": [640, 480],
                "framerate": 30,
                "jpeg_quality": 85,
                "rotation": 0,
            },
            "audio": {
                "device": "default",
                "volume": 80,
                "sounds_dir": os.path.join(self._temp_dir, "sounds"),
            },
            "status_leds": {
                "pin": 18,
                "count": 8,
                "brightness": 50,
            },
            "sensors": {
                "bme680": {"enabled": True, "i2c_address": "0x77"},
                "scd40": {"enabled": True, "i2c_address": "0x62"},
                "sht45": {"enabled": True, "i2c_address": "0x44"},
            },
            "intelligence": {
                "model_dir": os.path.join(self._temp_dir, "models"),
                "confidence_threshold": 0.60,
                "motion_threshold": 5.0,
            },
            "heads": {
                "active_head": None,
                "detection": {
                    "enabled": True,
                    "eeprom_address": "0x50",
                    "gpio_pins": [20, 21, 22],
                },
            },
            "mission": {
                "missions_dir": os.path.join(self._temp_dir, "missions"),
                "max_duration_minutes": 120,
            },
            "network": {"wifi_check_interval_s": 30},
            "health": {
                "interval_s": 10,
                "cpu_warn_pct": 80,
                "cpu_critical_pct": 95,
                "temp_warn_c": 70,
                "temp_critical_c": 80,
                "disk_warn_pct": 80,
                "disk_critical_pct": 95,
            },
            "streaming": {"port": 8080, "quality": 85},
            "obstacle": {
                "enabled": True,
                "poll_interval": 0.05,
                "sensors": {
                    "front": {"trigger_pin": 23, "echo_pin": 24},
                    "left_front": {"trigger_pin": 25, "echo_pin": 8},
                    "right_front": {"trigger_pin": 7, "echo_pin": 1},
                },
            },
            "path_planner": {
                "cell_size_m": 0.5,
                "obstacle_buffer": 1,
                "max_iterations": 10000,
                "unknown_penalty": 1.5,
                "simplify_paths": True,
                "simplify_tolerance_m": 1.0,
            },
            "behavior": {
                "tick_rate_hz": 10,
                "emergency_stop_cooldown_s": 5.0,
            },
            "learning": {
                "enabled": True,
                "interval_hours": 1,
                "min_samples_calibration": 10,
                "min_samples_patrol": 5,
                "anomaly_sigma_threshold": 3.0,
                "baseline_hours": 72,
            },
        }

        config_path: str = os.path.join(self._temp_dir, "cerberus.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(config, f, default_flow_style=False)

        return config_path

    def _print_header(self) -> None:
        print("\n" + "=" * 60)
        print("  CERBERUS AUTONOMOUS AI ROVER — SMOKE TEST")
        print("  Operation Ground Truth")
        print("=" * 60)

    def _print_info(self, msg: str) -> None:
        print(f"  ℹ  {msg}")

    def _print_result(self, result: TestResult) -> None:
        icon: str = "✓" if result.passed else "✗"
        status: str = "PASS" if result.passed else "FAIL"
        line: str = f"  {icon} [{status}] {result.name} ({result.duration_ms:.0f}ms)"
        if result.passed:
            line += f" — {result.message}"
        else:
            line += f" — {result.error}"
        print(line)

    def _print_fatal(self, msg: str) -> None:
        print(f"\n  ✗ FATAL: {msg}")

    def _print_summary(self) -> None:
        r: SmokeResults = self._results
        print("\n" + "-" * 60)
        print(f"  Results: {r.passed}/{r.total} passed, "
              f"{r.failed} failed, "
              f"{r.duration_s:.2f}s total")

        if r.failed > 0:
            print("\n  Failed tests:")
            for t in r.results:
                if not t.passed:
                    print(f"    ✗ {t.name}: {t.error}")

        status: str = "ALL SYSTEMS GO" if r.all_passed else "FAILURES DETECTED"
        print(f"\n  {'🟢' if r.all_passed else '🔴'} {status}")
        print("=" * 60 + "\n")


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Cerberus Smoke Test"
    )
    parser.add_argument(
        "--config", "-c",
        type=str, default=None,
        help="Path to cerberus.yaml config file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full tracebacks on failure"
    )
    args: argparse.Namespace = parser.parse_args()

    tester: SmokeTester = SmokeTester(
        config_path=args.config,
        verbose=args.verbose
    )

    try:
        results: SmokeResults = tester.run_all()
        return 0 if results.all_passed else 1
    except Exception as e:
        print(f"\n  FATAL: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())