"""
Cerberus Head Detector
Identifies which physical payload head is connected to the rover.
Supports three detection methods in priority order:
    1. I2C ID chip (EEPROM with head identifier)
    2. GPIO pin combination (binary encoding via pull-up/pull-down)
    3. Config override (manual selection in cerberus.yaml)
Falls back gracefully — if hardware detection fails, config wins.
"""

import logging
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class HeadType(Enum):
    """Known payload head types."""
    NONE = "none"
    WEED_SCANNER = "weed_scanner"
    SURVEILLANCE = "surveillance"
    ENV_LOGGER = "env_logger"
    PEST_DETERRENT = "pest_deterrent"
    BIRD_WATCHER = "bird_watcher"
    MICROCLIMATE = "microclimate"
    UNKNOWN = "unknown"


class DetectionMethod(Enum):
    """How the head was identified."""
    I2C_EEPROM = "i2c_eeprom"
    GPIO_PINS = "gpio_pins"
    CONFIG_OVERRIDE = "config_override"
    NOT_DETECTED = "not_detected"


# GPIO pin encoding — 3 pins give 8 combinations (6 heads + none + unknown)
HEAD_GPIO_MAP: dict[tuple[int, int, int], HeadType] = {
    (0, 0, 0): HeadType.NONE,
    (0, 0, 1): HeadType.WEED_SCANNER,
    (0, 1, 0): HeadType.SURVEILLANCE,
    (0, 1, 1): HeadType.ENV_LOGGER,
    (1, 0, 0): HeadType.PEST_DETERRENT,
    (1, 0, 1): HeadType.BIRD_WATCHER,
    (1, 1, 0): HeadType.MICROCLIMATE,
    (1, 1, 1): HeadType.UNKNOWN,
}

# I2C EEPROM identifier bytes mapped to head types
HEAD_I2C_MAP: dict[int, HeadType] = {
    0x00: HeadType.NONE,
    0x01: HeadType.WEED_SCANNER,
    0x02: HeadType.SURVEILLANCE,
    0x03: HeadType.ENV_LOGGER,
    0x04: HeadType.PEST_DETERRENT,
    0x05: HeadType.BIRD_WATCHER,
    0x06: HeadType.MICROCLIMATE,
}

# Config string mapping
HEAD_NAME_MAP: dict[str, HeadType] = {
    "none": HeadType.NONE,
    "weed_scanner": HeadType.WEED_SCANNER,
    "surveillance": HeadType.SURVEILLANCE,
    "env_logger": HeadType.ENV_LOGGER,
    "pest_deterrent": HeadType.PEST_DETERRENT,
    "bird_watcher": HeadType.BIRD_WATCHER,
    "microclimate": HeadType.MICROCLIMATE,
}


@dataclass
class HeadDetectionResult:
    """Result of head detection attempt."""
    head_type: HeadType = HeadType.NONE
    method: DetectionMethod = DetectionMethod.NOT_DETECTED
    confidence: float = 0.0
    raw_value: Any = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_type": self.head_type.value,
            "method": self.method.value,
            "confidence": round(self.confidence, 2),
            "raw_value": str(self.raw_value) if self.raw_value is not None else None,
            "message": self.message
        }


class HeadDetector:
    """
    Detects the physically connected payload head.
    Checks I2C EEPROM first, then GPIO pins, then config override.
    First successful detection wins.

    Hardware setup for I2C EEPROM:
        - AT24C02 or similar small EEPROM on each head's connector
        - Address 0x50 (default for AT24C02)
        - Byte 0x00 contains the head type identifier

    Hardware setup for GPIO encoding:
        - Three GPIO pins with pull-down resistors on the rover
        - Each head has a pin header that connects specific pins to 3.3V
        - The combination of high/low across the three pins identifies the head

    Config override:
        - Set heads.active_head in cerberus.yaml to force a specific head
        - Useful for development and when hardware detection is not installed
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config

        self._eeprom_address: int = config.get(
            "heads", "detection", "eeprom_address", default=0x50
        )
        self._eeprom_bus: int = config.get(
            "heads", "detection", "eeprom_bus", default=1
        )
        self._gpio_pins: list[int] = config.get(
            "heads", "detection", "gpio_pins", default=[20, 21, 27]
        )
        self._config_override: Optional[str] = config.get(
            "heads", "active_head", default=None
        )
        self._detection_enabled: bool = config.get(
            "heads", "detection", "enabled", default=True
        )

        self._last_result: Optional[HeadDetectionResult] = None
        self._smbus: Optional[Any] = None
        self._gpio_available: bool = False

        self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize hardware interfaces for detection."""
        try:
            import smbus2
            self._smbus = smbus2.SMBus(self._eeprom_bus)
            logger.debug("I2C bus %d opened for head detection", self._eeprom_bus)
        except (ImportError, OSError):
            self._smbus = None
            logger.debug("I2C not available for head detection")

        try:
            import gpiozero
            self._gpio_available = True
            logger.debug("GPIO available for head detection")
        except ImportError:
            self._gpio_available = False
            logger.debug("GPIO not available for head detection")

    def detect(self) -> HeadDetectionResult:
        """
        Detect the connected head. Tries methods in priority order:
            1. Config override (if set, skip hardware)
            2. I2C EEPROM
            3. GPIO pins
            4. Default to config or NONE
        """
        if self._config_override:
            result: HeadDetectionResult = self._detect_from_config()
            self._last_result = result
            return result

        if not self._detection_enabled:
            result = HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.NOT_DETECTED,
                message="Hardware detection disabled in config"
            )
            self._last_result = result
            return result

        result = self._detect_from_i2c()
        if result.head_type not in (HeadType.NONE, HeadType.UNKNOWN):
            self._last_result = result
            logger.info("Head detected via I2C: %s", result.head_type.value)
            return result

        result = self._detect_from_gpio()
        if result.head_type not in (HeadType.NONE, HeadType.UNKNOWN):
            self._last_result = result
            logger.info("Head detected via GPIO: %s", result.head_type.value)
            return result

        result = HeadDetectionResult(
            head_type=HeadType.NONE,
            method=DetectionMethod.NOT_DETECTED,
            message="No head detected via hardware — check connection or set config override"
        )
        self._last_result = result
        logger.warning(result.message)
        return result

    def _detect_from_config(self) -> HeadDetectionResult:
        """Detect head from config override."""
        name: str = self._config_override.lower().strip()
        head_type: HeadType = HEAD_NAME_MAP.get(name, HeadType.UNKNOWN)

        if head_type == HeadType.UNKNOWN:
            logger.error("Invalid head name in config override: %s", name)
            return HeadDetectionResult(
                head_type=HeadType.UNKNOWN,
                method=DetectionMethod.CONFIG_OVERRIDE,
                confidence=0.0,
                raw_value=name,
                message=f"Unknown head name: {name}. Valid: {list(HEAD_NAME_MAP.keys())}"
            )

        logger.info("Head set via config override: %s", head_type.value)
        return HeadDetectionResult(
            head_type=head_type,
            method=DetectionMethod.CONFIG_OVERRIDE,
            confidence=1.0,
            raw_value=name,
            message=f"Config override: {name}"
        )

    def _detect_from_i2c(self) -> HeadDetectionResult:
        """Read head identifier from I2C EEPROM."""
        if self._smbus is None:
            return HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.I2C_EEPROM,
                message="I2C not available"
            )

        try:
            head_byte: int = self._smbus.read_byte_data(self._eeprom_address, 0x00)
            head_type: HeadType = HEAD_I2C_MAP.get(head_byte, HeadType.UNKNOWN)

            if head_type == HeadType.UNKNOWN:
                logger.warning("Unknown EEPROM head byte: 0x%02X", head_byte)

            return HeadDetectionResult(
                head_type=head_type,
                method=DetectionMethod.I2C_EEPROM,
                confidence=1.0 if head_type != HeadType.UNKNOWN else 0.0,
                raw_value=hex(head_byte),
                message=f"EEPROM byte: 0x{head_byte:02X} -> {head_type.value}"
            )

        except OSError as e:
            logger.debug("I2C EEPROM read failed (no head EEPROM?): %s", e)
            return HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.I2C_EEPROM,
                message=f"EEPROM not responding: {e}"
            )

    def _detect_from_gpio(self) -> HeadDetectionResult:
        """Read head identifier from GPIO pin combination."""
        if not self._gpio_available:
            return HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.GPIO_PINS,
                message="GPIO not available"
            )

        if len(self._gpio_pins) != 3:
            return HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.GPIO_PINS,
                message=f"Expected 3 GPIO pins, got {len(self._gpio_pins)}"
            )

        try:
            from gpiozero import InputDevice

            pin_values: list[int] = []
            for pin_num in self._gpio_pins:
                pin: InputDevice = InputDevice(pin_num, pull_up=False)
                pin_values.append(1 if pin.is_active else 0)
                pin.close()

            combo: tuple[int, int, int] = (pin_values[0], pin_values[1], pin_values[2])
            head_type: HeadType = HEAD_GPIO_MAP.get(combo, HeadType.UNKNOWN)

            return HeadDetectionResult(
                head_type=head_type,
                method=DetectionMethod.GPIO_PINS,
                confidence=1.0 if head_type not in (HeadType.NONE, HeadType.UNKNOWN) else 0.0,
                raw_value=combo,
                message=f"GPIO pins {self._gpio_pins}: {combo} -> {head_type.value}"
            )

        except Exception as e:
            logger.error("GPIO head detection failed: %s", e)
            return HeadDetectionResult(
                head_type=HeadType.NONE,
                method=DetectionMethod.GPIO_PINS,
                message=f"GPIO read error: {e}"
            )

    @property
    def last_result(self) -> Optional[HeadDetectionResult]:
        return self._last_result

    def get_head_class(self, head_type: HeadType) -> Optional[type]:
        """Return the head class for a given head type. Lazy import to avoid circular deps."""
        if head_type == HeadType.WEED_SCANNER:
            from cerberus.heads.weed_scanner import WeedScannerHead
            return WeedScannerHead
        elif head_type == HeadType.SURVEILLANCE:
            from cerberus.heads.surveillance import SurveillanceHead
            return SurveillanceHead
        elif head_type == HeadType.ENV_LOGGER:
            from cerberus.heads.env_logger import EnvLoggerHead
            return EnvLoggerHead
        elif head_type == HeadType.PEST_DETERRENT:
            from cerberus.heads.pest_deterrent import PestDeterrentHead
            return PestDeterrentHead
        elif head_type == HeadType.BIRD_WATCHER:
            from cerberus.heads.bird_watcher import BirdWatcherHead
            return BirdWatcherHead
        elif head_type == HeadType.MICROCLIMATE:
            from cerberus.heads.microclimate import MicroclimateHead
            return MicroclimateHead
        else:
            return None

    def detect_and_instantiate(self, config: Optional[CerberusConfig] = None) -> Optional[Any]:
        """Detect head and return an instantiated head object."""
        result: HeadDetectionResult = self.detect()

        if result.head_type in (HeadType.NONE, HeadType.UNKNOWN):
            logger.warning("No head to instantiate — detection result: %s", result.head_type.value)
            return None

        head_class: Optional[type] = self.get_head_class(result.head_type)
        if head_class is None:
            logger.error("No class found for head type: %s", result.head_type.value)
            return None

        try:
            head: Any = head_class(config or self._config)
            logger.info("Head instantiated: %s via %s", result.head_type.value, result.method.value)
            return head
        except Exception as e:
            logger.error("Failed to instantiate %s: %s", result.head_type.value, e)
            return None

    def close(self) -> None:
        """Release hardware resources."""
        if self._smbus is not None:
            try:
                self._smbus.close()
            except Exception:
                pass
            self._smbus = None