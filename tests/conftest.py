"""
Cerberus Test Fixtures
Shared fixtures for all test modules. Creates minimal valid config,
temporary database, and mock subsystems so every module can be tested
in simulation mode without hardware.
"""

import os
import sys
import tempfile
import shutil
import logging
import pytest
from typing import Any, Generator

import yaml


PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


MINIMAL_CONFIG: dict[str, Any] = {
    "system": {
        "name": "cerberus-test",
        "version": "1.0.0-test",
        "environment": "development",
        "log_level": "DEBUG",
        "log_dir": "data/logs",
        "heartbeat_interval": 10,
        "main_loop_interval": 1.0
    },
    "database": {
        "path": "data/cerberus_test.db",
        "journal_mode": "WAL",
        "busy_timeout_ms": 5000,
        "max_log_age_days": 30
    },
    "i2c": {
        "bus": 1
    },
    "mqtt": {
        "broker": "localhost",
        "port": 1883,
        "keepalive": 60,
        "qos": 1,
        "topics": {
            "health": "cerberus/telemetry/health",
            "sensors": "cerberus/telemetry/sensors",
            "detections": "cerberus/detections",
            "mission": "cerberus/mission/status",
            "commands": "cerberus/command",
            "alerts": "cerberus/alerts"
        }
    },
    "safety": {
        "poll_interval": 2,
        "battery_warn_pct": 25,
        "battery_critical_pct": 15,
        "battery_shutdown_pct": 10,
        "thermal_warn_c": 70,
        "thermal_critical_c": 80,
        "thermal_shutdown_c": 85,
        "motor_overcurrent_a": 10,
        "watchdog_timeout_s": 30
    },
    "power": {
        "i2c_address": "0x40",
        "channels": {
            "battery": {
                "channel": 0,
                "shunt_resistance_ohms": 0.01,
                "voltage_min": 9.6,
                "voltage_max": 12.6,
                "current_limit_a": 15
            },
            "motors": {
                "channel": 1,
                "shunt_resistance_ohms": 0.01,
                "current_limit_a": 10
            },
            "accessories": {
                "channel": 2,
                "shunt_resistance_ohms": 0.1,
                "current_limit_a": 3
            }
        }
    },
    "motors": {
        "left": {
            "pwm_pin": 12,
            "forward_pin": 5,
            "reverse_pin": 6
        },
        "right": {
            "pwm_pin": 13,
            "forward_pin": 16,
            "reverse_pin": 26
        },
        "frequency_hz": 1000,
        "emergency_stop_pin": 25
    },
    "servos": {
        "i2c_address": "0x40",
        "frequency_hz": 50,
        "channels": {
            "pan": {"channel": 0, "min_pulse_us": 500, "max_pulse_us": 2500},
            "tilt": {"channel": 1, "min_pulse_us": 500, "max_pulse_us": 2500}
        }
    },
    "navigation": {
        "home_lat": 36.1699,
        "home_lon": -115.1398,
        "home_radius_m": 1.5,
        "waypoint_radius_m": 2.0,
        "heading_tolerance_deg": 15,
        "spin_threshold_deg": 90,
        "gps_timeout_s": 10,
        "rtb_approach_speed": 0.3,
        "cruise_speed": 0.5,
        "rtb_max_retries": 3
    },
    "camera": {
        "still_width": 4608,
        "still_height": 2592,
        "stream_width": 1280,
        "stream_height": 720,
        "inference_width": 640,
        "inference_height": 480,
        "framerate": 30,
        "format": "BGR888",
        "jpeg_quality": 85
    },
    "audio": {
        "sample_rate": 48000,
        "channels": 1,
        "volume": 0.7,
        "audio_dir": "data/audio/predator"
    },
    "status_leds": {
        "pin": 18,
        "count": 12,
        "brightness": 0.3,
        "order": "GRB"
    },
    "sensors": {
        "bme680": {
            "enabled": True,
            "i2c_address": "0x77",
            "poll_interval": 10
        },
        "scd40": {
            "enabled": True,
            "i2c_address": "0x62",
            "poll_interval": 30
        },
        "sht45": {
            "enabled": True,
            "i2c_address": "0x44",
            "poll_interval": 5
        }
    },
    "intelligence": {
        "model_dir": "models",
        "default_threshold": 0.5,
        "motion": {
            "min_area": 500,
            "threshold": 25,
            "blur_size": 21,
            "history": 500,
            "var_threshold": 40,
            "detect_shadows": True,
            "dilate_iterations": 3,
            "erode_iterations": 1,
            "cooldown_seconds": 2.0
        }
    },
    "heads": {
        "active_head": "surveillance",
        "detection": {
            "enabled": False,
            "eeprom_address": "0x50",
            "eeprom_bus": 1,
            "gpio_pins": [20, 21, 27]
        },
        "weed_scanner": {"cycle_interval": 2.0, "confidence_threshold": 0.6},
        "surveillance": {"cycle_interval": 0.2, "confidence_threshold": 0.5},
        "env_logger": {"cycle_interval": 60.0},
        "pest_deterrent": {
            "cycle_interval": 0.5,
            "confidence_threshold": 0.5,
            "cooldown_seconds": 15.0,
            "effectiveness_window_seconds": 30.0
        },
        "bird_watcher": {
            "cycle_interval": 1.0,
            "confidence_threshold": 0.5,
            "audio_record_interval_seconds": 300.0,
            "audio_duration_seconds": 15.0,
            "sighting_cooldown_seconds": 60.0
        },
        "microclimate": {
            "cycle_interval": 5.0,
            "probe_height_cm": 5.0,
            "stabilization_seconds": 3.0,
            "samples_per_point": 5,
            "sample_interval_seconds": 1.0,
            "hotspot_threshold_c": 3.0,
            "coldspot_threshold_c": -3.0,
            "moisture_threshold_pct": 15.0
        }
    },
    "mission": {
        "missions_dir": "config/missions",
        "max_duration_minutes": 60,
        "patrol_dwell_seconds": 10.0,
        "patrol_speed": 0.4,
        "grid_spacing_m": 2.0,
        "grid_speed": 0.3
    },
    "network": {
        "wifi_interface": "wlan0"
    },
    "health": {
        "poll_interval": 5,
        "cpu_warn_pct": 80,
        "cpu_critical_pct": 95,
        "memory_warn_pct": 80,
        "memory_critical_pct": 95,
        "disk_warn_pct": 85,
        "disk_critical_pct": 95,
        "temp_warn_c": 70,
        "temp_critical_c": 80,
        "temp_shutdown_c": 85
    },
    "streaming": {
        "host": "0.0.0.0",
        "port": 8080,
        "framerate": 15,
        "jpeg_quality": 70
    },
    "obstacle": {
        "enabled": True,
        "poll_interval": 0.1,
        "median_samples": 3,
        "clear_threshold_cm": 100.0,
        "warning_threshold_cm": 60.0,
        "caution_threshold_cm": 30.0,
        "warning_speed_limit": 0.5,
        "caution_speed_limit": 0.0,
        "front": {"trigger_pin": 17, "echo_pin": 27},
        "left_front": {"trigger_pin": 22, "echo_pin": 10},
        "right_front": {"trigger_pin": 9, "echo_pin": 11}
    },
    "path_planner": {
        "cell_size_m": 0.5,
        "unknown_penalty": 1.5,
        "max_iterations": 10000,
        "simplify_paths": True,
        "simplify_tolerance_m": 1.0
    },
    "behavior": {
        "threat_confidence_min": 0.7,
        "investigation_confidence_min": 0.6,
        "investigation_cooldown_s": 30.0
    },
    "learning": {
        "interval_seconds": 3600,
        "min_samples_calibration": 20,
        "min_samples_patrol": 10,
        "anomaly_sigma_threshold": 2.0,
        "baseline_hours": 168,
        "zone_radius_deg": 0.00005
    }
}


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test artifacts."""
    path: str = tempfile.mkdtemp(prefix="cerberus_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def config_path(temp_dir: str) -> str:
    """Write minimal config to a temp YAML file and return the path."""
    config: dict[str, Any] = MINIMAL_CONFIG.copy()
    config["database"]["path"] = os.path.join(temp_dir, "cerberus_test.db")
    config["system"]["log_dir"] = os.path.join(temp_dir, "logs")

    yaml_path: str = os.path.join(temp_dir, "cerberus.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    return yaml_path


@pytest.fixture
def config(config_path: str) -> Generator:
    """Create a CerberusConfig instance from the temp config file."""
    from cerberus.core.config import CerberusConfig
    CerberusConfig.reset()
    cfg = CerberusConfig(config_path)
    yield cfg
    CerberusConfig.reset()


@pytest.fixture
def db(config) -> Generator:
    """Create a CerberusDB instance with test database."""
    from cerberus.storage.db import CerberusDB
    CerberusDB.reset()
    database = CerberusDB(config)
    yield database
    database.close_all()
    CerberusDB.reset()