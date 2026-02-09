"""
Cerberus Environmental Sensor Interface
Reads temperature, humidity, pressure, gas resistance, CO2, and VOC
from BME680, SCD-40, and SHT45 sensors over I2C.
Each sensor gracefully degrades independently — one sensor failure
does not take down the others.
"""

import time
import logging
import threading
from typing import Any, Optional
from dataclasses import dataclass

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class EnvironmentReading:
    """Combined reading from all environmental sensors."""
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    pressure_hpa: float = 0.0
    gas_resistance_ohms: float = 0.0
    co2_ppm: float = 0.0
    voc_index: float = 0.0
    probe_temperature_c: float = 0.0
    probe_humidity_pct: float = 0.0
    bme680_ok: bool = False
    scd40_ok: bool = False
    sht45_ok: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "pressure_hpa": round(self.pressure_hpa, 1),
            "gas_resistance_ohms": round(self.gas_resistance_ohms, 0),
            "co2_ppm": round(self.co2_ppm, 0),
            "voc_index": round(self.voc_index, 0),
            "probe_temperature_c": round(self.probe_temperature_c, 1),
            "probe_humidity_pct": round(self.probe_humidity_pct, 1),
            "bme680_ok": self.bme680_ok,
            "scd40_ok": self.scd40_ok,
            "sht45_ok": self.sht45_ok,
            "timestamp": self.timestamp
        }


class BME680Sensor:
    """
    BME680 — temperature, humidity, pressure, gas resistance (VOC proxy).
    I2C address: 0x77
    """

    def __init__(self, config: CerberusConfig) -> None:
        self._enabled: bool = config.get("sensors", "bme680", "enabled", default=True)
        self._address: int = config.get("sensors", "bme680", "address", default=0x77)
        self._sensor: Optional[Any] = None
        self._available: bool = False

        if self._enabled:
            self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize BME680 sensor."""
        try:
            import board
            import adafruit_bme680

            i2c = board.I2C()
            self._sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=self._address)
            self._sensor.sea_level_pressure = 1013.25
            self._available = True
            logger.info("BME680 initialized at 0x%02X", self._address)

        except ImportError:
            logger.warning("BME680 library not available — sensor disabled (dev mode)")
            self._available = False

        except Exception as e:
            logger.error("BME680 initialization failed: %s", e)
            self._available = False

    def read(self) -> dict[str, float]:
        """Read all BME680 values. Returns dict with sensor data."""
        if not self._available or self._sensor is None:
            return {
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "pressure_hpa": 0.0,
                "gas_resistance_ohms": 0.0,
                "ok": False
            }

        try:
            return {
                "temperature_c": float(self._sensor.temperature),
                "humidity_pct": float(self._sensor.relative_humidity),
                "pressure_hpa": float(self._sensor.pressure),
                "gas_resistance_ohms": float(self._sensor.gas),
                "ok": True
            }
        except Exception as e:
            logger.error("BME680 read error: %s", e)
            return {
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "pressure_hpa": 0.0,
                "gas_resistance_ohms": 0.0,
                "ok": False
            }

    @property
    def available(self) -> bool:
        return self._available


class SCD40Sensor:
    """
    SCD-40 — CO2 concentration, temperature, humidity.
    I2C address: 0x62
    """

    def __init__(self, config: CerberusConfig) -> None:
        self._enabled: bool = config.get("sensors", "scd40", "enabled", default=True)
        self._address: int = config.get("sensors", "scd40", "address", default=0x62)
        self._sensor: Optional[Any] = None
        self._available: bool = False

        if self._enabled:
            self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize SCD-40 sensor."""
        try:
            import board
            import adafruit_scd4x

            i2c = board.I2C()
            self._sensor = adafruit_scd4x.SCD4X(i2c, address=self._address)
            self._sensor.start_periodic_measurement()
            self._available = True
            logger.info("SCD-40 initialized at 0x%02X — periodic measurement started", self._address)

        except ImportError:
            logger.warning("SCD-40 library not available — sensor disabled (dev mode)")
            self._available = False

        except Exception as e:
            logger.error("SCD-40 initialization failed: %s", e)
            self._available = False

    def read(self) -> dict[str, float]:
        """Read CO2 and climate data from SCD-40."""
        if not self._available or self._sensor is None:
            return {
                "co2_ppm": 0.0,
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "ok": False
            }

        try:
            if not self._sensor.data_ready:
                return {
                    "co2_ppm": 0.0,
                    "temperature_c": 0.0,
                    "humidity_pct": 0.0,
                    "ok": False
                }

            return {
                "co2_ppm": float(self._sensor.CO2),
                "temperature_c": float(self._sensor.temperature),
                "humidity_pct": float(self._sensor.relative_humidity),
                "ok": True
            }
        except Exception as e:
            logger.error("SCD-40 read error: %s", e)
            return {
                "co2_ppm": 0.0,
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "ok": False
            }

    def stop_measurement(self) -> None:
        """Stop SCD-40 periodic measurement. Called during shutdown."""
        if self._available and self._sensor is not None:
            try:
                self._sensor.stop_periodic_measurement()
                logger.info("SCD-40 periodic measurement stopped")
            except Exception as e:
                logger.error("SCD-40 stop error: %s", e)

    @property
    def available(self) -> bool:
        return self._available


class SHT45Sensor:
    """
    SHT45 — high-precision temperature and humidity probe.
    Used on the extendable arm for microclimate mapping.
    I2C address: 0x44
    """

    def __init__(self, config: CerberusConfig) -> None:
        self._enabled: bool = config.get("sensors", "sht45", "enabled", default=True)
        self._address: int = config.get("sensors", "sht45", "address", default=0x44)
        self._sensor: Optional[Any] = None
        self._available: bool = False

        if self._enabled:
            self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize SHT45 sensor."""
        try:
            import board
            from adafruit_sht4x import SHT4x

            i2c = board.I2C()
            self._sensor = SHT4x(i2c, address=self._address)
            self._available = True
            logger.info("SHT45 initialized at 0x%02X", self._address)

        except ImportError:
            logger.warning("SHT45 library not available — sensor disabled (dev mode)")
            self._available = False

        except Exception as e:
            logger.error("SHT45 initialization failed: %s", e)
            self._available = False

    def read(self) -> dict[str, float]:
        """Read temperature and humidity from SHT45."""
        if not self._available or self._sensor is None:
            return {
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "ok": False
            }

        try:
            temperature: float = float(self._sensor.temperature)
            humidity: float = float(self._sensor.relative_humidity)

            return {
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "ok": True
            }
        except Exception as e:
            logger.error("SHT45 read error: %s", e)
            return {
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "ok": False
            }

    @property
    def available(self) -> bool:
        return self._available


class EnvironmentManager:
    """
    Manages all environmental sensors as a unified interface.
    Polls sensors on a configurable interval and provides
    combined readings. Individual sensor failures are isolated.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._poll_interval: float = config.get("sensors", "poll_interval_seconds", default=10)

        self._bme680: BME680Sensor = BME680Sensor(config)
        self._scd40: SCD40Sensor = SCD40Sensor(config)
        self._sht45: SHT45Sensor = SHT45Sensor(config)

        self._latest: EnvironmentReading = EnvironmentReading()
        self._lock: threading.Lock = threading.Lock()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._reading_callbacks: list = []
        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None

        available: list[str] = []
        if self._bme680.available:
            available.append("BME680")
        if self._scd40.available:
            available.append("SCD-40")
        if self._sht45.available:
            available.append("SHT45")

        status: str = ", ".join(available) if available else "none (simulation mode)"
        logger.info("Environment manager created — sensors: %s, poll=%.0fs", status, self._poll_interval)

    def bind_db(self, db: Any) -> None:
        """Bind database for sensor logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for telemetry publishing."""
        self._mqtt = mqtt_client

    def register_callback(self, callback) -> None:
        """Register callback for new readings: callback(reading)."""
        self._reading_callbacks.append(callback)

    def take_reading(self) -> EnvironmentReading:
        """Take an immediate reading from all sensors."""
        bme_data: dict[str, float] = self._bme680.read()
        scd_data: dict[str, float] = self._scd40.read()
        sht_data: dict[str, float] = self._sht45.read()

        reading: EnvironmentReading = EnvironmentReading(
            temperature_c=bme_data["temperature_c"],
            humidity_pct=bme_data["humidity_pct"],
            pressure_hpa=bme_data["pressure_hpa"],
            gas_resistance_ohms=bme_data["gas_resistance_ohms"],
            co2_ppm=scd_data["co2_ppm"],
            voc_index=bme_data["gas_resistance_ohms"],
            probe_temperature_c=sht_data["temperature_c"],
            probe_humidity_pct=sht_data["humidity_pct"],
            bme680_ok=bme_data.get("ok", False),
            scd40_ok=scd_data.get("ok", False),
            sht45_ok=sht_data.get("ok", False),
            timestamp=time.time()
        )

        if not reading.bme680_ok and reading.scd40_ok:
            reading.temperature_c = scd_data["temperature_c"]
            reading.humidity_pct = scd_data["humidity_pct"]

        with self._lock:
            self._latest = reading

        self._persist(reading)
        self._publish(reading)
        self._notify(reading)

        return reading

    def _persist(self, reading: EnvironmentReading) -> None:
        """Log reading to database."""
        if self._db is None:
            return
        try:
            self._db.log_sensor(reading.to_dict())
        except Exception as e:
            logger.error("Failed to persist environment reading: %s", e)

    def _publish(self, reading: EnvironmentReading) -> None:
        """Publish reading to MQTT."""
        if self._mqtt is None:
            return
        try:
            self._mqtt.publish_sensor(reading.to_dict())
        except Exception as e:
            logger.error("Failed to publish environment reading: %s", e)

    def _notify(self, reading: EnvironmentReading) -> None:
        """Call registered callbacks with new reading."""
        for cb in self._reading_callbacks:
            try:
                cb(reading)
            except Exception as e:
                logger.error("Environment reading callback error: %s", e)

    def _poll_loop(self) -> None:
        """Background polling loop for continuous sensor readings."""
        logger.info("Environment polling started — interval=%.0fs", self._poll_interval)

        while self._running:
            try:
                reading: EnvironmentReading = self.take_reading()

                if not self._any_sensor_available():
                    logger.debug(
                        "Environment [SIM]: temp=%.1f°C, hum=%.1f%%, co2=%.0fppm",
                        reading.temperature_c, reading.humidity_pct, reading.co2_ppm
                    )

            except Exception as e:
                logger.error("Environment poll error: %s", e)

            for _ in range(int(self._poll_interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("Environment polling stopped")

    def _any_sensor_available(self) -> bool:
        """Check if any physical sensor is connected."""
        return self._bme680.available or self._scd40.available or self._sht45.available

    def start(self) -> None:
        """Start background sensor polling."""
        if self._running:
            logger.warning("Environment manager already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="environment-poller",
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sensor polling and release resources."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 2)
            if self._thread.is_alive():
                logger.warning("Environment thread did not stop cleanly")
            self._thread = None

        logger.info("Environment polling stopped")

    @property
    def latest(self) -> EnvironmentReading:
        """Get the most recent environment reading. Thread-safe."""
        with self._lock:
            return self._latest

    @property
    def bme680(self) -> BME680Sensor:
        return self._bme680

    @property
    def scd40(self) -> SCD40Sensor:
        return self._scd40

    @property
    def sht45(self) -> SHT45Sensor:
        return self._sht45

    @property
    def is_running(self) -> bool:
        return self._running

    def release(self) -> None:
        """Release all sensor resources. Called during shutdown."""
        self.stop()
        self._scd40.stop_measurement()
        logger.info("Environment manager resources released")

    def __repr__(self) -> str:
        sensors: list[str] = []
        if self._bme680.available:
            sensors.append("BME680")
        if self._scd40.available:
            sensors.append("SCD-40")
        if self._sht45.available:
            sensors.append("SHT45")

        return (
            f"EnvironmentManager(sensors=[{', '.join(sensors) or 'sim'}], "
            f"running={self._running})"
        )