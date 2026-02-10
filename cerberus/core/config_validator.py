"""
Cerberus Configuration Validator
Validates cerberus.yaml at startup before any subsystem boots. Checks
required fields, types, value ranges, GPIO pin conflicts, I2C address
conflicts, and cross-references between subsystems. Bad config never
reaches runtime.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class Severity(Enum):
    """Validation issue severity."""
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation issue found in the config."""
    path: str
    message: str
    severity: Severity = Severity.ERROR
    value: Any = None

    def __str__(self) -> str:
        prefix: str = "ERROR" if self.severity == Severity.ERROR else "WARN"
        val_str: str = f" (got: {self.value})" if self.value is not None else ""
        return f"[{prefix}] {self.path}: {self.message}{val_str}"


@dataclass
class ValidationResult:
    """Complete validation result."""
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def add_error(self, path: str, message: str, value: Any = None) -> None:
        self.issues.append(ValidationIssue(path, message, Severity.ERROR, value))

    def add_warning(self, path: str, message: str, value: Any = None) -> None:
        self.issues.append(ValidationIssue(path, message, Severity.WARNING, value))

    def summary(self) -> str:
        if self.valid and self.warning_count == 0:
            return "Configuration valid — no issues found"
        parts: list[str] = []
        if self.error_count > 0:
            parts.append(f"{self.error_count} error(s)")
        if self.warning_count > 0:
            parts.append(f"{self.warning_count} warning(s)")
        return f"Configuration validation: {', '.join(parts)}"


class ConfigValidator:
    """
    Validates the full cerberus.yaml configuration file.
    Call validate() at startup before any subsystem initializes.
    Returns a ValidationResult with all issues found.
    """

    VALID_GPIO_PINS: set[int] = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
    }

    VALID_I2C_ADDRESSES: set[int] = set(range(0x03, 0x78))

    VALID_LOG_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    VALID_ENVIRONMENTS: set[str] = {"development", "production"}

    VALID_ESCALATIONS: set[str] = {"passive", "gentle", "moderate", "aggressive"}

    VALID_THREAT_LEVELS: set[str] = {"none", "low", "medium", "high", "critical"}

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()
        self._config: CerberusConfig = config
        self._result: ValidationResult = ValidationResult()
        self._used_gpio: dict[int, str] = {}
        self._used_i2c: dict[int, str] = {}

    def validate(self) -> ValidationResult:
        """Run all validation checks. Returns the complete result."""
        self._result = ValidationResult()
        self._used_gpio = {}
        self._used_i2c = {}

        self._validate_system()
        self._validate_database()
        self._validate_mqtt()
        self._validate_health()
        self._validate_safety()
        self._validate_power()
        self._validate_motors()
        self._validate_drive()
        self._validate_navigation()
        self._validate_gps()
        self._validate_camera()
        self._validate_streaming()
        self._validate_servos()
        self._validate_leds()
        self._validate_audio()
        self._validate_sensors()
        self._validate_intelligence()
        self._validate_mission()
        self._validate_heads()
        self._check_gpio_conflicts()
        self._check_i2c_conflicts()

        for issue in self._result.issues:
            if issue.severity == Severity.ERROR:
                logger.error(str(issue))
            else:
                logger.warning(str(issue))

        logger.info(self._result.summary())
        return self._result

    def _get(self, *keys: str, default: Any = None) -> Any:
        """Safe config access."""
        return self._config.get(*keys, default=default)

    def _require_field(self, path: str, *keys: str) -> Any:
        """Check that a required field exists."""
        value: Any = self._get(*keys)
        if value is None:
            self._result.add_error(path, "Required field missing")
            return None
        return value

    def _check_type(self, path: str, value: Any, expected_type: type) -> bool:
        """Validate value type."""
        if value is None:
            return True
        if not isinstance(value, expected_type):
            self._result.add_error(path, f"Expected {expected_type.__name__}, got {type(value).__name__}", value)
            return False
        return True

    def _check_range(self, path: str, value: Any, min_val: float, max_val: float) -> bool:
        """Validate numeric value is within range."""
        if value is None:
            return True
        if not isinstance(value, (int, float)):
            return False
        if value < min_val or value > max_val:
            self._result.add_error(path, f"Must be between {min_val} and {max_val}", value)
            return False
        return True

    def _check_positive(self, path: str, value: Any) -> bool:
        """Validate value is positive."""
        if value is None:
            return True
        if not isinstance(value, (int, float)) or value <= 0:
            self._result.add_error(path, "Must be positive", value)
            return False
        return True

    def _check_non_negative(self, path: str, value: Any) -> bool:
        """Validate value is zero or positive."""
        if value is None:
            return True
        if not isinstance(value, (int, float)) or value < 0:
            self._result.add_error(path, "Must be non-negative", value)
            return False
        return True

    def _check_in_set(self, path: str, value: Any, valid: set) -> bool:
        """Validate value is in a set of valid options."""
        if value is None:
            return True
        if value not in valid:
            self._result.add_error(path, f"Must be one of: {sorted(valid)}", value)
            return False
        return True

    def _register_gpio(self, path: str, pin: Any) -> None:
        """Register a GPIO pin for conflict detection."""
        if pin is None or not isinstance(pin, int):
            return
        if pin not in self.VALID_GPIO_PINS:
            self._result.add_error(path, f"Invalid GPIO pin number", pin)
            return
        self._used_gpio.setdefault(pin, path)

    def _register_i2c(self, path: str, address: Any) -> None:
        """Register an I2C address for conflict detection."""
        if address is None:
            return
        if isinstance(address, str):
            try:
                address = int(address, 16)
            except ValueError:
                self._result.add_error(path, "Invalid I2C address format", address)
                return
        if address not in self.VALID_I2C_ADDRESSES:
            self._result.add_error(path, "I2C address out of valid range (0x03-0x77)", hex(address))
            return
        self._used_i2c.setdefault(address, path)

    def _check_gpio_conflicts(self) -> None:
        """Check for GPIO pins assigned to multiple functions."""
        pin_users: dict[int, list[str]] = {}
        for pin, path in self._used_gpio.items():
            pin_users.setdefault(pin, []).append(path)

        for pin, users in pin_users.items():
            if len(users) > 1:
                self._result.add_error(
                    "gpio_conflicts",
                    f"GPIO {pin} assigned to multiple functions: {', '.join(users)}"
                )

    def _check_i2c_conflicts(self) -> None:
        """Check for I2C addresses assigned to multiple devices."""
        addr_users: dict[int, list[str]] = {}
        for addr, path in self._used_i2c.items():
            addr_users.setdefault(addr, []).append(path)

        for addr, users in addr_users.items():
            if len(users) > 1:
                self._result.add_error(
                    "i2c_conflicts",
                    f"I2C address {hex(addr)} assigned to multiple devices: {', '.join(users)}"
                )

    def _validate_system(self) -> None:
        """Validate system section."""
        self._require_field("system.name", "system", "name")

        log_level: Any = self._get("system", "log_level", default="INFO")
        self._check_in_set("system.log_level", log_level, self.VALID_LOG_LEVELS)

        env: Any = self._get("system", "environment", default="development")
        self._check_in_set("system.environment", env, self.VALID_ENVIRONMENTS)

        heartbeat: Any = self._get("system", "heartbeat_interval")
        if heartbeat is not None:
            self._check_positive("system.heartbeat_interval", heartbeat)

        loop_interval: Any = self._get("system", "main_loop_interval")
        if loop_interval is not None:
            self._check_range("system.main_loop_interval", loop_interval, 0.1, 60.0)

    def _validate_database(self) -> None:
        """Validate database section."""
        self._require_field("database.path", "database", "path")

        busy: Any = self._get("database", "busy_timeout_ms")
        if busy is not None:
            self._check_range("database.busy_timeout_ms", busy, 100, 30000)

        age: Any = self._get("database", "max_log_age_days")
        if age is not None:
            self._check_range("database.max_log_age_days", age, 1, 365)

    def _validate_mqtt(self) -> None:
        """Validate MQTT section."""
        self._require_field("mqtt.broker", "mqtt", "broker")

        port: Any = self._get("mqtt", "port", default=1883)
        self._check_range("mqtt.port", port, 1, 65535)

        keepalive: Any = self._get("mqtt", "keepalive")
        if keepalive is not None:
            self._check_range("mqtt.keepalive", keepalive, 5, 600)

        qos: Any = self._get("mqtt", "qos", default=1)
        self._check_in_set("mqtt.qos", qos, {0, 1, 2})

    def _validate_health(self) -> None:
        """Validate health monitoring thresholds."""
        self._check_positive("health.poll_interval", self._get("health", "poll_interval"))

        cpu_warn: Any = self._get("health", "cpu_warn_pct")
        cpu_crit: Any = self._get("health", "cpu_critical_pct")
        if cpu_warn is not None and cpu_crit is not None:
            self._check_range("health.cpu_warn_pct", cpu_warn, 1, 100)
            self._check_range("health.cpu_critical_pct", cpu_crit, 1, 100)
            if cpu_warn >= cpu_crit:
                self._result.add_error("health", "cpu_warn_pct must be less than cpu_critical_pct")

        temp_warn: Any = self._get("health", "temp_warn_c")
        temp_crit: Any = self._get("health", "temp_critical_c")
        temp_shut: Any = self._get("health", "temp_shutdown_c")
        if temp_warn and temp_crit and temp_shut:
            if not (temp_warn < temp_crit < temp_shut):
                self._result.add_error("health", "Temperature thresholds must be: warn < critical < shutdown")

    def _validate_safety(self) -> None:
        """Validate safety watchdog thresholds."""
        batt_warn: Any = self._get("safety", "battery_warn_pct")
        batt_crit: Any = self._get("safety", "battery_critical_pct")
        batt_shut: Any = self._get("safety", "battery_shutdown_pct")

        if batt_warn and batt_crit and batt_shut:
            if not (batt_shut < batt_crit < batt_warn):
                self._result.add_error("safety", "Battery thresholds must be: shutdown < critical < warn")

        thermal_warn: Any = self._get("safety", "thermal_warn_c")
        thermal_crit: Any = self._get("safety", "thermal_critical_c")
        thermal_shut: Any = self._get("safety", "thermal_shutdown_c")

        if thermal_warn and thermal_crit and thermal_shut:
            if not (thermal_warn < thermal_crit < thermal_shut):
                self._result.add_error("safety", "Thermal thresholds must be: warn < critical < shutdown")

        overcurrent: Any = self._get("safety", "motor_overcurrent_a")
        if overcurrent is not None:
            self._check_range("safety.motor_overcurrent_a", overcurrent, 0.1, 50.0)

    def _validate_power(self) -> None:
        """Validate power monitoring section."""
        i2c_addr: Any = self._get("power", "i2c_address")
        if i2c_addr is not None:
            self._register_i2c("power.ina3221", i2c_addr)

        for ch_name in ["battery", "motors", "accessories"]:
            shunt: Any = self._get("power", "channels", ch_name, "shunt_resistance_ohms")
            if shunt is not None:
                self._check_range(f"power.channels.{ch_name}.shunt_resistance_ohms", shunt, 0.001, 10.0)

        vmin: Any = self._get("power", "channels", "battery", "voltage_min")
        vmax: Any = self._get("power", "channels", "battery", "voltage_max")
        if vmin is not None and vmax is not None:
            if vmin >= vmax:
                self._result.add_error("power.channels.battery", "voltage_min must be less than voltage_max")
            self._check_range("power.channels.battery.voltage_min", vmin, 5.0, 20.0)
            self._check_range("power.channels.battery.voltage_max", vmax, 5.0, 20.0)

    def _validate_motors(self) -> None:
        """Validate motor configuration and GPIO pins."""
        for side in ["left", "right"]:
            pwm: Any = self._get("motors", side, "pwm_pin")
            fwd: Any = self._get("motors", side, "forward_pin")
            rev: Any = self._get("motors", side, "reverse_pin")

            if pwm is not None:
                self._register_gpio(f"motors.{side}.pwm_pin", pwm)
            else:
                self._result.add_error(f"motors.{side}.pwm_pin", "Required field missing")

            if fwd is not None:
                self._register_gpio(f"motors.{side}.forward_pin", fwd)
            else:
                self._result.add_error(f"motors.{side}.forward_pin", "Required field missing")

            if rev is not None:
                self._register_gpio(f"motors.{side}.reverse_pin", rev)
            else:
                self._result.add_error(f"motors.{side}.reverse_pin", "Required field missing")

        freq: Any = self._get("motors", "frequency_hz")
        if freq is not None:
            self._check_range("motors.frequency_hz", freq, 100, 50000)

        estop: Any = self._get("motors", "emergency_stop_pin")
        if estop is not None:
            self._register_gpio("motors.emergency_stop_pin", estop)

    def _validate_drive(self) -> None:
        """Validate drive controller settings."""
        for key in ["default_speed", "turn_speed", "spin_speed", "max_speed"]:
            val: Any = self._get("drive", key)
            if val is not None:
                self._check_range(f"drive.{key}", val, 0.0, 1.0)

    def _validate_navigation(self) -> None:
        """Validate navigation settings."""
        home_lat: Any = self._get("navigation", "home_lat")
        home_lon: Any = self._get("navigation", "home_lon")

        if home_lat is not None:
            self._check_range("navigation.home_lat", home_lat, -90.0, 90.0)
        if home_lon is not None:
            self._check_range("navigation.home_lon", home_lon, -180.0, 180.0)

        home_radius: Any = self._get("navigation", "home_radius_m")
        if home_radius is not None:
            self._check_range("navigation.home_radius_m", home_radius, 0.5, 20.0)

        heading_tol: Any = self._get("navigation", "heading_tolerance_deg")
        if heading_tol is not None:
            self._check_range("navigation.heading_tolerance_deg", heading_tol, 1.0, 90.0)

        rtb_retries: Any = self._get("navigation", "rtb_max_retries")
        if rtb_retries is not None:
            self._check_range("navigation.rtb_max_retries", rtb_retries, 0, 20)

    def _validate_gps(self) -> None:
        """Validate GPS settings."""
        port: Any = self._get("gps", "port")
        if port is not None:
            self._check_range("gps.port", port, 1, 65535)

        min_sats: Any = self._get("gps", "min_satellites")
        if min_sats is not None:
            self._check_range("gps.min_satellites", min_sats, 1, 24)

        max_hdop: Any = self._get("gps", "max_hdop")
        if max_hdop is not None:
            self._check_range("gps.max_hdop", max_hdop, 0.5, 50.0)

    def _validate_camera(self) -> None:
        """Validate camera settings."""
        for key in ["still_width", "still_height", "stream_width", "stream_height",
                     "inference_width", "inference_height"]:
            val: Any = self._get("camera", key)
            if val is not None:
                self._check_range(f"camera.{key}", val, 64, 8192)

        fps: Any = self._get("camera", "framerate")
        if fps is not None:
            self._check_range("camera.framerate", fps, 1, 120)

        quality: Any = self._get("camera", "jpeg_quality")
        if quality is not None:
            self._check_range("camera.jpeg_quality", quality, 1, 100)

    def _validate_streaming(self) -> None:
        """Validate MJPEG streaming settings."""
        port: Any = self._get("streaming", "port")
        if port is not None:
            self._check_range("streaming.port", port, 1024, 65535)

        mqtt_port: Any = self._get("mqtt", "port")
        if port is not None and mqtt_port is not None and port == mqtt_port:
            self._result.add_error("streaming.port", "Conflicts with MQTT port", port)

    def _validate_servos(self) -> None:
        """Validate servo/PCA9685 settings."""
        i2c_addr: Any = self._get("servos", "i2c_address")
        if i2c_addr is not None:
            self._register_i2c("servos.pca9685", i2c_addr)

        freq: Any = self._get("servos", "frequency_hz")
        if freq is not None:
            self._check_range("servos.frequency_hz", freq, 24, 1526)

        channels: Any = self._get("servos", "channels")
        if channels and isinstance(channels, dict):
            used_channels: dict[int, str] = {}
            for name, cfg in channels.items():
                if not isinstance(cfg, dict):
                    continue
                ch: Any = cfg.get("channel")
                if ch is not None:
                    self._check_range(f"servos.channels.{name}.channel", ch, 0, 15)
                    if ch in used_channels:
                        self._result.add_error(
                            f"servos.channels.{name}",
                            f"PCA9685 channel {ch} conflicts with {used_channels[ch]}"
                        )
                    else:
                        used_channels[ch] = name

                min_pulse: Any = cfg.get("min_pulse_us")
                max_pulse: Any = cfg.get("max_pulse_us")
                if min_pulse is not None and max_pulse is not None:
                    if min_pulse >= max_pulse:
                        self._result.add_error(
                            f"servos.channels.{name}",
                            "min_pulse_us must be less than max_pulse_us"
                        )

    def _validate_leds(self) -> None:
        """Validate LED settings."""
        status_pin: Any = self._get("leds", "status", "pin")
        if status_pin is not None:
            self._register_gpio("leds.status.pin", status_pin)

        count: Any = self._get("leds", "status", "count")
        if count is not None:
            self._check_range("leds.status.count", count, 1, 300)

        brightness: Any = self._get("leds", "status", "brightness")
        if brightness is not None:
            self._check_range("leds.status.brightness", brightness, 0.0, 1.0)

        ir_pin: Any = self._get("leds", "ir", "pin")
        if ir_pin is not None:
            self._register_gpio("leds.ir.pin", ir_pin)

        eyes_pin: Any = self._get("leds", "pest_eyes", "pin")
        if eyes_pin is not None:
            self._register_gpio("leds.pest_eyes.pin", eyes_pin)

    def _validate_audio(self) -> None:
        """Validate audio settings."""
        volume: Any = self._get("audio", "volume")
        if volume is not None:
            self._check_range("audio.volume", volume, 0.0, 1.0)

        sample_rate: Any = self._get("audio", "sample_rate")
        if sample_rate is not None:
            self._check_in_set("audio.sample_rate", sample_rate, {8000, 16000, 22050, 44100, 48000})

    def _validate_sensors(self) -> None:
        """Validate environmental sensor settings."""
        for sensor_name, default_addr in [("bme680", 0x77), ("scd40", 0x62), ("sht45", 0x44)]:
            enabled: Any = self._get("sensors", sensor_name, "enabled")
            if enabled is False:
                continue

            addr: Any = self._get("sensors", sensor_name, "i2c_address")
            if addr is not None:
                self._register_i2c(f"sensors.{sensor_name}", addr)

            poll: Any = self._get("sensors", sensor_name, "poll_interval")
            if poll is not None:
                self._check_range(f"sensors.{sensor_name}.poll_interval", poll, 1.0, 3600.0)

    def _validate_intelligence(self) -> None:
        """Validate AI/ML settings."""
        min_area: Any = self._get("intelligence", "motion", "min_area")
        if min_area is not None:
            self._check_range("intelligence.motion.min_area", min_area, 10, 100000)

        threshold: Any = self._get("intelligence", "motion", "threshold")
        if threshold is not None:
            self._check_range("intelligence.motion.threshold", threshold, 1, 255)

        blur: Any = self._get("intelligence", "motion", "blur_size")
        if blur is not None:
            if not isinstance(blur, int) or blur % 2 == 0:
                self._result.add_error("intelligence.motion.blur_size", "Must be an odd integer", blur)

        default_thresh: Any = self._get("intelligence", "default_threshold")
        if default_thresh is not None:
            self._check_range("intelligence.default_threshold", default_thresh, 0.0, 1.0)

    def _validate_mission(self) -> None:
        """Validate mission and autonomy settings."""
        max_dur: Any = self._get("mission", "max_duration_minutes")
        if max_dur is not None:
            self._check_range("mission.max_duration_minutes", max_dur, 1, 1440)

        grid_spacing: Any = self._get("mission", "grid_spacing_m")
        if grid_spacing is not None:
            self._check_range("mission.grid_spacing_m", grid_spacing, 0.25, 50.0)

    def _validate_heads(self) -> None:
        """Validate all payload head configurations."""
        head_names: list[str] = [
            "weed_scanner", "surveillance", "env_logger",
            "pest_deterrent", "bird_watcher", "microclimate"
        ]

        for head in head_names:
            cycle: Any = self._get("heads", head, "cycle_interval")
            if cycle is not None:
                self._check_range(f"heads.{head}.cycle_interval", cycle, 0.1, 3600.0)

            threshold: Any = self._get("heads", head, "confidence_threshold")
            if threshold is not None:
                self._check_range(f"heads.{head}.confidence_threshold", threshold, 0.0, 1.0)

        samples: Any = self._get("heads", "microclimate", "samples_per_point")
        if samples is not None:
            self._check_range("heads.microclimate.samples_per_point", samples, 1, 50)

        stab: Any = self._get("heads", "microclimate", "stabilization_seconds")
        if stab is not None:
            self._check_range("heads.microclimate.stabilization_seconds", stab, 0.0, 60.0)

        cooldown: Any = self._get("heads", "pest_deterrent", "cooldown_seconds")
        if cooldown is not None:
            self._check_range("heads.pest_deterrent.cooldown_seconds", cooldown, 0.0, 300.0)

        eff_window: Any = self._get("heads", "pest_deterrent", "effectiveness_window_seconds")
        if eff_window is not None:
            self._check_range("heads.pest_deterrent.effectiveness_window_seconds", eff_window, 5.0, 300.0)

        audio_interval: Any = self._get("heads", "bird_watcher", "audio_record_interval_seconds")
        if audio_interval is not None:
            self._check_range("heads.bird_watcher.audio_record_interval_seconds", audio_interval, 30.0, 3600.0)


def validate_config(config: Optional[CerberusConfig] = None) -> ValidationResult:
    """Convenience function to validate the configuration."""
    validator: ConfigValidator = ConfigValidator(config)
    return validator.validate()
