"""
Tests for cerberus.heads.head_detector — HeadDetector
Validates detection methods (config override, I2C, GPIO), head type
mapping, head instantiation, and fallback behavior. All hardware
is mocked since tests run on dev machine.
"""

import pytest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from cerberus.core.config import CerberusConfig
from cerberus.heads.head_detector import (
    HeadDetector, HeadDetectionResult, HeadType, DetectionMethod,
    HEAD_NAME_MAP, HEAD_I2C_MAP, HEAD_GPIO_MAP
)


class TestHeadType:
    """HeadType enum coverage."""

    def test_all_head_types_exist(self) -> None:
        expected: list[str] = [
            "none", "weed_scanner", "surveillance", "env_logger",
            "pest_deterrent", "bird_watcher", "microclimate", "unknown"
        ]
        for name in expected:
            assert HeadType(name) is not None

    def test_name_map_covers_all_heads(self) -> None:
        real_heads: list[HeadType] = [
            h for h in HeadType if h not in (HeadType.UNKNOWN,)
        ]
        for head in real_heads:
            assert head.value in HEAD_NAME_MAP or head == HeadType.NONE


class TestDetectionMethod:
    """DetectionMethod enum coverage."""

    def test_all_methods_exist(self) -> None:
        expected: list[str] = ["i2c_eeprom", "gpio_pins", "config_override", "not_detected"]
        for name in expected:
            assert DetectionMethod(name) is not None


class TestHeadDetectionResult:
    """HeadDetectionResult dataclass."""

    def test_default_result(self) -> None:
        result: HeadDetectionResult = HeadDetectionResult()
        assert result.head_type == HeadType.NONE
        assert result.method == DetectionMethod.NOT_DETECTED
        assert result.confidence == 0.0

    def test_to_dict(self) -> None:
        result: HeadDetectionResult = HeadDetectionResult(
            head_type=HeadType.SURVEILLANCE,
            method=DetectionMethod.CONFIG_OVERRIDE,
            confidence=1.0,
            raw_value="surveillance",
            message="Config override"
        )
        d: dict[str, Any] = result.to_dict()
        assert d["head_type"] == "surveillance"
        assert d["method"] == "config_override"
        assert d["confidence"] == 1.0
        assert d["raw_value"] == "surveillance"

    def test_to_dict_none_raw_value(self) -> None:
        result: HeadDetectionResult = HeadDetectionResult()
        d: dict[str, Any] = result.to_dict()
        assert d["raw_value"] is None


class TestConfigOverrideDetection:
    """Head detection via config override (active_head set)."""

    def test_detect_surveillance_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "surveillance"
        config._data["heads"]["detection"]["enabled"] = False
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.SURVEILLANCE
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        assert result.confidence == 1.0
        detector.close()

    def test_detect_weed_scanner_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "weed_scanner"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.WEED_SCANNER
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        detector.close()

    def test_detect_env_logger_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "env_logger"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.ENV_LOGGER
        detector.close()

    def test_detect_pest_deterrent_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "pest_deterrent"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.PEST_DETERRENT
        detector.close()

    def test_detect_bird_watcher_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "bird_watcher"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.BIRD_WATCHER
        detector.close()

    def test_detect_microclimate_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "microclimate"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.MICROCLIMATE
        detector.close()

    def test_detect_none_override(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "none"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.NONE
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        detector.close()

    def test_invalid_override_returns_unknown(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "flamethrower"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.UNKNOWN
        assert result.confidence == 0.0
        detector.close()

    def test_override_case_insensitive(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "SURVEILLANCE"
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.SURVEILLANCE
        detector.close()

    def test_override_with_whitespace(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "  bird_watcher  "
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.BIRD_WATCHER
        detector.close()


class TestNoHardwareDetection:
    """Fallback when no hardware and no override."""

    def test_no_override_no_hardware(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        config._data["heads"]["detection"]["enabled"] = True
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.NONE
        assert result.method == DetectionMethod.NOT_DETECTED
        detector.close()

    def test_detection_disabled(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        config._data["heads"]["detection"]["enabled"] = False
        detector: HeadDetector = HeadDetector(config)
        result: HeadDetectionResult = detector.detect()
        assert result.head_type == HeadType.NONE
        assert "disabled" in result.message.lower()
        detector.close()


class TestI2CDetection:
    """Head detection via mocked I2C EEPROM."""

    def test_i2c_detect_weed_scanner(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        config._data["heads"]["detection"]["enabled"] = True

        mock_smbus: MagicMock = MagicMock()
        mock_smbus.read_byte_data.return_value = 0x01

        detector: HeadDetector = HeadDetector(config)
        detector._smbus = mock_smbus
        detector._hardware_available = False

        result: HeadDetectionResult = detector._detect_from_i2c()
        assert result.head_type == HeadType.WEED_SCANNER
        assert result.method == DetectionMethod.I2C_EEPROM
        assert result.confidence == 1.0
        detector.close()

    def test_i2c_detect_all_known_bytes(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)

        for byte_val, expected_type in HEAD_I2C_MAP.items():
            mock_smbus: MagicMock = MagicMock()
            mock_smbus.read_byte_data.return_value = byte_val
            detector._smbus = mock_smbus

            result: HeadDetectionResult = detector._detect_from_i2c()
            assert result.head_type == expected_type, (
                f"Byte 0x{byte_val:02X} expected {expected_type} got {result.head_type}"
            )

        detector.close()

    def test_i2c_unknown_byte(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)

        mock_smbus: MagicMock = MagicMock()
        mock_smbus.read_byte_data.return_value = 0xFF
        detector._smbus = mock_smbus

        result: HeadDetectionResult = detector._detect_from_i2c()
        assert result.head_type == HeadType.UNKNOWN
        assert result.confidence == 0.0
        detector.close()

    def test_i2c_oserror_returns_none(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)

        mock_smbus: MagicMock = MagicMock()
        mock_smbus.read_byte_data.side_effect = OSError("No device")
        detector._smbus = mock_smbus

        result: HeadDetectionResult = detector._detect_from_i2c()
        assert result.head_type == HeadType.NONE
        detector.close()

    def test_i2c_not_available(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)
        detector._smbus = None

        result: HeadDetectionResult = detector._detect_from_i2c()
        assert result.head_type == HeadType.NONE
        assert "not available" in result.message
        detector.close()


class TestGPIODetection:
    """Head detection via mocked GPIO pins."""

    def test_gpio_detect_all_combos(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)
        detector._gpio_available = True

        for combo, expected_type in HEAD_GPIO_MAP.items():
            mock_devices: list[MagicMock] = []
            for val in combo:
                dev: MagicMock = MagicMock()
                dev.is_active = bool(val)
                dev.close = MagicMock()
                mock_devices.append(dev)

            with patch("cerberus.heads.head_detector.InputDevice", side_effect=mock_devices):
                result: HeadDetectionResult = detector._detect_from_gpio()
                assert result.head_type == expected_type, (
                    f"Combo {combo} expected {expected_type} got {result.head_type}"
                )

        detector.close()

    def test_gpio_not_available(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)
        detector._gpio_available = False

        result: HeadDetectionResult = detector._detect_from_gpio()
        assert result.head_type == HeadType.NONE
        assert "not available" in result.message
        detector.close()

    def test_gpio_wrong_pin_count(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        config._data["heads"]["detection"]["gpio_pins"] = [20, 21]
        detector: HeadDetector = HeadDetector(config)
        detector._gpio_available = True

        result: HeadDetectionResult = detector._detect_from_gpio()
        assert result.head_type == HeadType.NONE
        assert "Expected 3" in result.message
        detector.close()

    def test_gpio_exception_returns_none(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        detector: HeadDetector = HeadDetector(config)
        detector._gpio_available = True

        with patch("cerberus.heads.head_detector.InputDevice", side_effect=RuntimeError("GPIO fail")):
            result: HeadDetectionResult = detector._detect_from_gpio()
            assert result.head_type == HeadType.NONE

        detector.close()


class TestDetectionPriority:
    """Detection method priority: config > I2C > GPIO."""

    def test_config_override_skips_hardware(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "surveillance"
        detector: HeadDetector = HeadDetector(config)

        mock_smbus: MagicMock = MagicMock()
        mock_smbus.read_byte_data.return_value = 0x01
        detector._smbus = mock_smbus

        result: HeadDetectionResult = detector.detect()
        assert result.method == DetectionMethod.CONFIG_OVERRIDE
        assert result.head_type == HeadType.SURVEILLANCE
        mock_smbus.read_byte_data.assert_not_called()
        detector.close()

    def test_i2c_checked_before_gpio(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = None
        config._data["heads"]["detection"]["enabled"] = True
        detector: HeadDetector = HeadDetector(config)

        mock_smbus: MagicMock = MagicMock()
        mock_smbus.read_byte_data.return_value = 0x03
        detector._smbus = mock_smbus
        detector._gpio_available = True

        result: HeadDetectionResult = detector.detect()
        assert result.method == DetectionMethod.I2C_EEPROM
        assert result.head_type == HeadType.ENV_LOGGER
        detector.close()


class TestGetHeadClass:
    """Head class mapping for instantiation."""

    def test_returns_none_for_none_type(self, config: CerberusConfig) -> None:
        detector: HeadDetector = HeadDetector(config)
        assert detector.get_head_class(HeadType.NONE) is None
        detector.close()

    def test_returns_none_for_unknown_type(self, config: CerberusConfig) -> None:
        detector: HeadDetector = HeadDetector(config)
        assert detector.get_head_class(HeadType.UNKNOWN) is None
        detector.close()

    def test_returns_class_for_each_head(self, config: CerberusConfig) -> None:
        detector: HeadDetector = HeadDetector(config)
        heads_with_classes: list[HeadType] = [
            HeadType.WEED_SCANNER, HeadType.SURVEILLANCE, HeadType.ENV_LOGGER,
            HeadType.PEST_DETERRENT, HeadType.BIRD_WATCHER, HeadType.MICROCLIMATE
        ]
        for head_type in heads_with_classes:
            cls = detector.get_head_class(head_type)
            assert cls is not None, f"No class for {head_type}"
        detector.close()


class TestLastResult:
    """Result caching."""

    def test_last_result_updated_after_detect(self, config: CerberusConfig) -> None:
        config._data["heads"]["active_head"] = "surveillance"
        detector: HeadDetector = HeadDetector(config)
        assert detector.last_result is None

        detector.detect()
        assert detector.last_result is not None
        assert detector.last_result.head_type == HeadType.SURVEILLANCE
        detector.close()


class TestClose:
    """Resource cleanup."""

    def test_close_releases_smbus(self, config: CerberusConfig) -> None:
        detector: HeadDetector = HeadDetector(config)
        mock_smbus: MagicMock = MagicMock()
        detector._smbus = mock_smbus

        detector.close()
        mock_smbus.close.assert_called_once()
        assert detector._smbus is None

    def test_close_handles_smbus_error(self, config: CerberusConfig) -> None:
        detector: HeadDetector = HeadDetector(config)
        mock_smbus: MagicMock = MagicMock()
        mock_smbus.close.side_effect = RuntimeError("close failed")
        detector._smbus = mock_smbus

        detector.close()
        assert detector._smbus is None