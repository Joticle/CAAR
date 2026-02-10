"""
Tests for cerberus.core.config_validator — ConfigValidator
Validates required fields, type checks, range validation, GPIO/I2C
conflict detection, and cross-reference validation.
"""

import pytest
from typing import Any
from copy import deepcopy

from cerberus.core.config import CerberusConfig
from cerberus.core.config_validator import (
    ConfigValidator, ValidationResult, ValidationIssue, Severity,
    validate_config
)
from tests.conftest import MINIMAL_CONFIG


def make_config(overrides: dict[str, Any] = None) -> dict[str, Any]:
    """Create a config dict with optional overrides for testing."""
    cfg: dict[str, Any] = deepcopy(MINIMAL_CONFIG)
    if overrides:
        for dotted_key, value in overrides.items():
            keys: list[str] = dotted_key.split(".")
            target: dict[str, Any] = cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value
    return cfg


class TestValidationResult:
    """ValidationResult dataclass behavior."""

    def test_empty_result_is_valid(self) -> None:
        result: ValidationResult = ValidationResult()
        assert result.valid
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_add_error_makes_invalid(self) -> None:
        result: ValidationResult = ValidationResult()
        result.add_error("test.path", "something broke")
        assert not result.valid
        assert result.error_count == 1

    def test_add_warning_stays_valid(self) -> None:
        result: ValidationResult = ValidationResult()
        result.add_warning("test.path", "heads up")
        assert result.valid
        assert result.warning_count == 1

    def test_mixed_errors_and_warnings(self) -> None:
        result: ValidationResult = ValidationResult()
        result.add_error("a", "err1")
        result.add_error("b", "err2")
        result.add_warning("c", "warn1")
        assert result.error_count == 2
        assert result.warning_count == 1
        assert not result.valid

    def test_summary_contains_counts(self) -> None:
        result: ValidationResult = ValidationResult()
        result.add_error("a", "err")
        result.add_warning("b", "warn")
        summary: str = result.summary()
        assert "1 error" in summary
        assert "1 warning" in summary


class TestValidConfig:
    """Minimal valid config passes validation."""

    def test_minimal_config_valid(self, config: CerberusConfig) -> None:
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        assert result.valid, f"Expected valid config but got: {result.summary()}"

    def test_validate_config_convenience(self, config: CerberusConfig) -> None:
        result: ValidationResult = validate_config(config)
        assert result.valid


class TestSystemValidation:
    """System section validation."""

    def test_missing_system_name(self, config: CerberusConfig) -> None:
        config._data["system"].pop("name", None)
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        errors: list[str] = [i.path for i in result.issues if i.severity == Severity.ERROR]
        assert any("system.name" in e for e in errors)

    def test_invalid_log_level(self, config: CerberusConfig) -> None:
        config._data["system"]["log_level"] = "VERBOSE"
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        errors: list[str] = [i.message for i in result.issues if i.severity == Severity.ERROR]
        assert any("log_level" in e.lower() or "VERBOSE" in e for e in errors)

    def test_invalid_environment(self, config: CerberusConfig) -> None:
        config._data["system"]["environment"] = "staging"
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "environment" in i.path and i.severity == Severity.ERROR
            for i in result.issues
        )
        assert has_issue


class TestDatabaseValidation:
    """Database section validation."""

    def test_missing_database_path(self, config: CerberusConfig) -> None:
        config._data["database"].pop("path", None)
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        errors: list[str] = [i.path for i in result.issues if i.severity == Severity.ERROR]
        assert any("database.path" in e for e in errors)

    def test_invalid_busy_timeout(self, config: CerberusConfig) -> None:
        config._data["database"]["busy_timeout_ms"] = -100
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "busy_timeout" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestMQTTValidation:
    """MQTT section validation."""

    def test_missing_broker(self, config: CerberusConfig) -> None:
        config._data["mqtt"].pop("broker", None)
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        errors: list[str] = [i.path for i in result.issues if i.severity == Severity.ERROR]
        assert any("mqtt.broker" in e for e in errors)

    def test_port_out_of_range(self, config: CerberusConfig) -> None:
        config._data["mqtt"]["port"] = 70000
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "mqtt.port" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue

    def test_invalid_qos(self, config: CerberusConfig) -> None:
        config._data["mqtt"]["qos"] = 5
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "qos" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestSafetyValidation:
    """Safety thresholds and ordering."""

    def test_battery_thresholds_wrong_order(self, config: CerberusConfig) -> None:
        config._data["safety"]["battery_warn_pct"] = 10
        config._data["safety"]["battery_critical_pct"] = 20
        config._data["safety"]["battery_shutdown_pct"] = 30
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "battery" in i.path and i.severity == Severity.ERROR
            for i in result.issues
        )
        assert has_issue

    def test_thermal_thresholds_wrong_order(self, config: CerberusConfig) -> None:
        config._data["safety"]["thermal_warn_c"] = 85
        config._data["safety"]["thermal_critical_c"] = 80
        config._data["safety"]["thermal_shutdown_c"] = 70
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "thermal" in i.path and i.severity == Severity.ERROR
            for i in result.issues
        )
        assert has_issue


class TestMotorValidation:
    """Motor GPIO pin validation."""

    def test_missing_left_pwm_pin(self, config: CerberusConfig) -> None:
        config._data["motors"]["left"].pop("pwm_pin", None)
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        errors: list[str] = [i.path for i in result.issues if i.severity == Severity.ERROR]
        assert any("motors" in e and "pwm_pin" in e for e in errors)

    def test_invalid_frequency(self, config: CerberusConfig) -> None:
        config._data["motors"]["frequency_hz"] = 50
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "frequency" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestNavigationValidation:
    """Navigation coordinate validation."""

    def test_latitude_out_of_range(self, config: CerberusConfig) -> None:
        config._data["navigation"]["home_lat"] = 95.0
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "home_lat" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue

    def test_longitude_out_of_range(self, config: CerberusConfig) -> None:
        config._data["navigation"]["home_lon"] = -200.0
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "home_lon" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestCameraValidation:
    """Camera settings validation."""

    def test_framerate_out_of_range(self, config: CerberusConfig) -> None:
        config._data["camera"]["framerate"] = 0
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "framerate" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue

    def test_jpeg_quality_out_of_range(self, config: CerberusConfig) -> None:
        config._data["camera"]["jpeg_quality"] = 150
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "jpeg_quality" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestDriveValidation:
    """Drive speed validation."""

    def test_speed_over_one(self, config: CerberusConfig) -> None:
        if "drive" not in config._data:
            config._data["drive"] = {}
        config._data["drive"]["max_speed"] = 1.5
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "max_speed" in i.path for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_issue


class TestGPIOConflictDetection:
    """GPIO pin conflict detection across subsystems."""

    def test_duplicate_gpio_detected(self, config: CerberusConfig) -> None:
        config._data["motors"]["left"]["pwm_pin"] = 13
        config._data["motors"]["right"]["pwm_pin"] = 13
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_conflict: bool = any(
            "GPIO" in i.message and "conflict" in i.message.lower()
            for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_conflict

    def test_motor_led_pin_conflict(self, config: CerberusConfig) -> None:
        config._data["motors"]["left"]["pwm_pin"] = 18
        config._data["status_leds"]["pin"] = 18
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_conflict: bool = any(
            "GPIO" in i.message and ("conflict" in i.message.lower() or "18" in i.message)
            for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_conflict


class TestI2CConflictDetection:
    """I2C address conflict detection."""

    def test_duplicate_i2c_detected(self, config: CerberusConfig) -> None:
        config._data["sensors"]["bme680"]["i2c_address"] = "0x62"
        config._data["sensors"]["scd40"]["i2c_address"] = "0x62"
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_conflict: bool = any(
            "I2C" in i.message and "conflict" in i.message.lower()
            for i in result.issues if i.severity == Severity.ERROR
        )
        assert has_conflict


class TestStreamingValidation:
    """Streaming port validation."""

    def test_streaming_port_conflict_with_mqtt(self, config: CerberusConfig) -> None:
        config._data["streaming"]["port"] = 1883
        config._data["mqtt"]["port"] = 1883
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_conflict: bool = any(
            "port" in i.message.lower() and ("conflict" in i.message.lower() or "same" in i.message.lower())
            for i in result.issues
        )
        assert has_conflict


class TestHealthValidation:
    """Health monitor threshold validation."""

    def test_cpu_warn_exceeds_critical(self, config: CerberusConfig) -> None:
        config._data["health"]["cpu_warn_pct"] = 98
        config._data["health"]["cpu_critical_pct"] = 90
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "cpu" in i.path.lower() and i.severity == Severity.ERROR
            for i in result.issues
        )
        assert has_issue

    def test_temp_warn_exceeds_critical(self, config: CerberusConfig) -> None:
        config._data["health"]["temp_warn_c"] = 85
        config._data["health"]["temp_critical_c"] = 75
        validator: ConfigValidator = ConfigValidator(config)
        result: ValidationResult = validator.validate()
        has_issue: bool = any(
            "temp" in i.path.lower() and i.severity == Severity.ERROR
            for i in result.issues
        )
        assert has_issue