"""
Cerberus Environmental Logger Head
BME680/SCD-40 sensor interface with autonomous interval logging
to SQLite. Self-generates daily trend analysis and publishes
environmental telemetry. Designed for long-duration stationary
monitoring or slow patrol environmental surveys.
"""

import time
import logging
import statistics
from typing import Any, Optional
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig
from cerberus.heads.base_head import BaseHead, HeadInfo


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class EnvironmentSnapshot:
    """A single environmental reading with analysis metadata."""
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    pressure_hpa: float = 0.0
    gas_resistance_ohms: float = 0.0
    co2_ppm: float = 0.0
    voc_index: float = 0.0
    heat_index_c: float = 0.0
    dew_point_c: float = 0.0
    comfort_level: str = "unknown"
    lat: float = 0.0
    lon: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": round(self.temperature_c, 2),
            "humidity_pct": round(self.humidity_pct, 2),
            "pressure_hpa": round(self.pressure_hpa, 2),
            "gas_resistance_ohms": round(self.gas_resistance_ohms, 0),
            "co2_ppm": round(self.co2_ppm, 0),
            "voc_index": round(self.voc_index, 1),
            "heat_index_c": round(self.heat_index_c, 1),
            "dew_point_c": round(self.dew_point_c, 1),
            "comfort_level": self.comfort_level,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "timestamp": self.timestamp
        }


@dataclass
class TrendAnalysis:
    """Statistical trend analysis over a time window."""
    window_name: str = ""
    sample_count: int = 0
    temp_min: float = 0.0
    temp_max: float = 0.0
    temp_avg: float = 0.0
    temp_trend: str = "stable"
    humidity_min: float = 0.0
    humidity_max: float = 0.0
    humidity_avg: float = 0.0
    pressure_avg: float = 0.0
    pressure_trend: str = "stable"
    co2_avg: float = 0.0
    co2_max: float = 0.0
    comfort_summary: str = ""
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_name": self.window_name,
            "sample_count": self.sample_count,
            "temp_min": round(self.temp_min, 1),
            "temp_max": round(self.temp_max, 1),
            "temp_avg": round(self.temp_avg, 1),
            "temp_trend": self.temp_trend,
            "humidity_min": round(self.humidity_min, 1),
            "humidity_max": round(self.humidity_max, 1),
            "humidity_avg": round(self.humidity_avg, 1),
            "pressure_avg": round(self.pressure_avg, 1),
            "pressure_trend": self.pressure_trend,
            "co2_avg": round(self.co2_avg, 0),
            "co2_max": round(self.co2_max, 0),
            "comfort_summary": self.comfort_summary,
            "generated_at": self.generated_at
        }


class EnvLoggerHead(BaseHead):
    """
    Head 3: Environmental Logger
    Reads BME680/SCD-40 sensors at configurable intervals, computes
    derived values (heat index, dew point, comfort level), logs to
    SQLite, publishes to MQTT, and generates trend analysis.

    Cycle:
        1. Read environmental sensors
        2. Compute derived values (heat index, dew point, comfort)
        3. Geotag with GPS
        4. Log snapshot to database
        5. Publish to MQTT telemetry
        6. Periodically generate trend analysis
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="env_logger",
        description="Autonomous environmental monitoring with trend analysis",
        version="1.0",
        requires_camera=False,
        requires_gps=True,
        requires_sensors=True,
        requires_audio=False,
        requires_servos=False,
        supported_tasks=["scan", "station_keep", "grid_scan"]
    )

    COMFORT_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
        "comfortable": {"temp": (18.0, 28.0), "humidity": (30.0, 60.0)},
        "warm": {"temp": (28.0, 35.0), "humidity": (0.0, 100.0)},
        "hot": {"temp": (35.0, 42.0), "humidity": (0.0, 100.0)},
        "extreme_heat": {"temp": (42.0, 100.0), "humidity": (0.0, 100.0)},
        "cool": {"temp": (10.0, 18.0), "humidity": (0.0, 100.0)},
        "cold": {"temp": (-40.0, 10.0), "humidity": (0.0, 100.0)},
    }

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._log_interval: float = self._config.get(
            "heads", "env_logger", "log_interval_seconds", default=60.0
        )
        self._trend_interval: float = self._config.get(
            "heads", "env_logger", "trend_interval_seconds", default=3600.0
        )
        self._trend_window: int = self._config.get(
            "heads", "env_logger", "trend_window_samples", default=60
        )
        self._alert_temp_high: float = self._config.get(
            "heads", "env_logger", "alert_temp_high_c", default=45.0
        )
        self._alert_temp_low: float = self._config.get(
            "heads", "env_logger", "alert_temp_low_c", default=0.0
        )
        self._alert_co2_high: float = self._config.get(
            "heads", "env_logger", "alert_co2_high_ppm", default=2000.0
        )
        self._alert_humidity_high: float = self._config.get(
            "heads", "env_logger", "alert_humidity_high_pct", default=85.0
        )

        self._snapshots: list[EnvironmentSnapshot] = []
        self._max_snapshots: int = 1440
        self._total_readings: int = 0
        self._alerts_sent: int = 0
        self._last_trend_time: float = 0.0
        self._latest_trend: Optional[TrendAnalysis] = None

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def _on_load(self) -> bool:
        """Verify environmental sensors are available."""
        if self._environment is None:
            logger.warning("No environmental sensors bound — env logger will simulate")

        logger.info(
            "Environmental logger configured — interval=%.0fs, trend_interval=%.0fs",
            self._log_interval, self._trend_interval
        )
        return True

    def _on_start(self) -> bool:
        """Start environmental monitoring."""
        self._total_readings = 0
        self._alerts_sent = 0
        self._last_trend_time = time.time()

        self._cycle_interval = self._log_interval

        logger.info("Environmental logger activated")
        return True

    def _on_stop(self) -> None:
        """Stop environmental monitoring."""
        if self._snapshots:
            self._generate_trend("session_final")

        logger.info(
            "Environmental logger deactivated — %d readings, %d alerts",
            self._total_readings, self._alerts_sent
        )

    def _on_unload(self) -> None:
        """Release env logger resources."""
        logger.info("Environmental logger unloaded")

    def _run_cycle(self) -> None:
        """One logging cycle: read, compute, log, publish, analyze."""
        snapshot: EnvironmentSnapshot = self._take_snapshot()
        self._total_readings += 1

        self._store_snapshot(snapshot)
        self._publish_snapshot(snapshot)
        self._check_alerts(snapshot)

        self._record_activity(
            f"Reading #{self._total_readings}: "
            f"{snapshot.temperature_c:.1f}°C, "
            f"{snapshot.humidity_pct:.0f}% RH, "
            f"{snapshot.co2_ppm:.0f} ppm CO2"
        )

        now: float = time.time()
        if now - self._last_trend_time >= self._trend_interval:
            self._generate_trend("hourly")
            self._last_trend_time = now

    def _take_snapshot(self) -> EnvironmentSnapshot:
        """Read all environmental sensors and compute derived values."""
        if self._environment is None:
            return self._simulated_snapshot()

        try:
            reading = self._environment.reading

            temp: float = reading.temperature_c
            humidity: float = reading.humidity_pct
            pressure: float = reading.pressure_hpa
            gas: float = reading.gas_resistance_ohms
            co2: float = reading.co2_ppm

            heat_index: float = self._compute_heat_index(temp, humidity)
            dew_point: float = self._compute_dew_point(temp, humidity)
            comfort: str = self._assess_comfort(temp, humidity, heat_index)

            gps_data: dict[str, Any] = self._get_gps_data()

            return EnvironmentSnapshot(
                temperature_c=temp,
                humidity_pct=humidity,
                pressure_hpa=pressure,
                gas_resistance_ohms=gas,
                co2_ppm=co2,
                heat_index_c=heat_index,
                dew_point_c=dew_point,
                comfort_level=comfort,
                lat=gps_data.get("lat", 0.0),
                lon=gps_data.get("lon", 0.0),
                timestamp=time.time()
            )

        except Exception as e:
            logger.error("Failed to read environmental sensors: %s", e)
            return EnvironmentSnapshot(timestamp=time.time())

    def _simulated_snapshot(self) -> EnvironmentSnapshot:
        """Generate simulated environmental data for dev environment."""
        import random
        import math

        hour: float = time.localtime().tm_hour + time.localtime().tm_min / 60.0
        temp_base: float = 35.0 + 10.0 * math.sin((hour - 6) * math.pi / 12.0)
        temp: float = temp_base + random.uniform(-1.0, 1.0)
        humidity: float = max(5.0, 25.0 - 5.0 * math.sin((hour - 6) * math.pi / 12.0) + random.uniform(-3.0, 3.0))
        pressure: float = 1013.25 + random.uniform(-2.0, 2.0)
        co2: float = 420.0 + random.uniform(-20.0, 40.0)

        heat_index: float = self._compute_heat_index(temp, humidity)
        dew_point: float = self._compute_dew_point(temp, humidity)
        comfort: str = self._assess_comfort(temp, humidity, heat_index)

        return EnvironmentSnapshot(
            temperature_c=temp,
            humidity_pct=humidity,
            pressure_hpa=pressure,
            gas_resistance_ohms=random.uniform(50000, 200000),
            co2_ppm=co2,
            heat_index_c=heat_index,
            dew_point_c=dew_point,
            comfort_level=comfort,
            lat=36.1699,
            lon=-115.1398,
            timestamp=time.time()
        )

    @staticmethod
    def _compute_heat_index(temp_c: float, humidity: float) -> float:
        """Compute heat index using Rothfusz regression (NWS formula)."""
        if temp_c < 27.0:
            return temp_c

        t: float = temp_c * 9.0 / 5.0 + 32.0
        rh: float = humidity

        hi: float = (
            -42.379
            + 2.04901523 * t
            + 10.14333127 * rh
            - 0.22475541 * t * rh
            - 0.00683783 * t * t
            - 0.05481717 * rh * rh
            + 0.00122874 * t * t * rh
            + 0.00085282 * t * rh * rh
            - 0.00000199 * t * t * rh * rh
        )

        return (hi - 32.0) * 5.0 / 9.0

    @staticmethod
    def _compute_dew_point(temp_c: float, humidity: float) -> float:
        """Compute dew point using Magnus formula."""
        import math

        if humidity <= 0:
            return temp_c

        a: float = 17.27
        b: float = 237.7

        alpha: float = (a * temp_c) / (b + temp_c) + math.log(humidity / 100.0)
        dew_point: float = (b * alpha) / (a - alpha)

        return dew_point

    def _assess_comfort(self, temp: float, humidity: float, heat_index: float) -> str:
        """Assess outdoor comfort level based on temperature and humidity."""
        effective_temp: float = max(temp, heat_index)

        for level, thresholds in self.COMFORT_THRESHOLDS.items():
            temp_range: tuple[float, float] = thresholds["temp"]
            humidity_range: tuple[float, float] = thresholds["humidity"]

            if temp_range[0] <= effective_temp < temp_range[1]:
                if humidity_range[0] <= humidity <= humidity_range[1]:
                    return level

        return "unknown"

    def _store_snapshot(self, snapshot: EnvironmentSnapshot) -> None:
        """Store snapshot in memory buffer and database."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

        if self._db is not None:
            try:
                self._db.log_sensor_reading(
                    sensor_type="environment",
                    data=snapshot.to_dict()
                )
            except Exception as e:
                logger.error("Failed to store env snapshot: %s", e)

    def _publish_snapshot(self, snapshot: EnvironmentSnapshot) -> None:
        """Publish snapshot to MQTT telemetry."""
        if self._mqtt is None:
            return

        try:
            self._mqtt.publish_sensor_data("environment", snapshot.to_dict())
        except Exception:
            pass

    def _check_alerts(self, snapshot: EnvironmentSnapshot) -> None:
        """Check for environmental alert conditions."""
        alerts: list[str] = []

        if snapshot.temperature_c >= self._alert_temp_high:
            alerts.append(f"EXTREME HEAT: {snapshot.temperature_c:.1f}°C")

        if snapshot.temperature_c <= self._alert_temp_low:
            alerts.append(f"FREEZING: {snapshot.temperature_c:.1f}°C")

        if snapshot.co2_ppm >= self._alert_co2_high:
            alerts.append(f"HIGH CO2: {snapshot.co2_ppm:.0f} ppm")

        if snapshot.humidity_pct >= self._alert_humidity_high:
            alerts.append(f"HIGH HUMIDITY: {snapshot.humidity_pct:.0f}%")

        for alert_msg in alerts:
            self._alerts_sent += 1
            logger.warning("ENV ALERT: %s", alert_msg)

            self._record_detection(
                detection_type="env_alert",
                label=alert_msg,
                confidence=1.0,
                metadata=snapshot.to_dict()
            )

    def _generate_trend(self, window_name: str) -> Optional[TrendAnalysis]:
        """Generate trend analysis from recent snapshots."""
        samples: list[EnvironmentSnapshot] = self._snapshots[-self._trend_window:]

        if len(samples) < 3:
            logger.debug("Not enough samples for trend analysis (%d)", len(samples))
            return None

        temps: list[float] = [s.temperature_c for s in samples]
        humids: list[float] = [s.humidity_pct for s in samples]
        pressures: list[float] = [s.pressure_hpa for s in samples if s.pressure_hpa > 0]
        co2s: list[float] = [s.co2_ppm for s in samples if s.co2_ppm > 0]

        temp_trend: str = self._calculate_trend(temps)
        pressure_trend: str = self._calculate_trend(pressures) if pressures else "stable"

        comfort_counts: dict[str, int] = {}
        for s in samples:
            comfort_counts[s.comfort_level] = comfort_counts.get(s.comfort_level, 0) + 1
        dominant_comfort: str = max(comfort_counts, key=comfort_counts.get) if comfort_counts else "unknown"

        trend: TrendAnalysis = TrendAnalysis(
            window_name=window_name,
            sample_count=len(samples),
            temp_min=min(temps),
            temp_max=max(temps),
            temp_avg=statistics.mean(temps),
            temp_trend=temp_trend,
            humidity_min=min(humids),
            humidity_max=max(humids),
            humidity_avg=statistics.mean(humids),
            pressure_avg=statistics.mean(pressures) if pressures else 0.0,
            pressure_trend=pressure_trend,
            co2_avg=statistics.mean(co2s) if co2s else 0.0,
            co2_max=max(co2s) if co2s else 0.0,
            comfort_summary=dominant_comfort,
            generated_at=time.time()
        )

        self._latest_trend = trend

        logger.info(
            "Trend [%s]: temp=%.1f-%.1f°C (%s), humidity=%.0f-%.0f%%, CO2=%.0f avg, comfort=%s",
            window_name,
            trend.temp_min, trend.temp_max, trend.temp_trend,
            trend.humidity_min, trend.humidity_max,
            trend.co2_avg, trend.comfort_summary
        )

        if self._mqtt is not None:
            try:
                self._mqtt.publish_sensor_data("env_trend", trend.to_dict())
            except Exception:
                pass

        if self._db is not None:
            try:
                self._db.log_sensor_reading(
                    sensor_type="env_trend",
                    data=trend.to_dict()
                )
            except Exception as e:
                logger.error("Failed to store trend analysis: %s", e)

        return trend

    @staticmethod
    def _calculate_trend(values: list[float]) -> str:
        """Determine trend direction from a series of values."""
        if len(values) < 3:
            return "stable"

        third: int = len(values) // 3
        first_avg: float = statistics.mean(values[:third])
        last_avg: float = statistics.mean(values[-third:])

        diff: float = last_avg - first_avg
        spread: float = max(values) - min(values)

        if spread == 0:
            return "stable"

        change_pct: float = abs(diff) / spread * 100

        if change_pct < 15:
            return "stable"
        elif diff > 0:
            return "rising"
        else:
            return "falling"

    def _get_gps_data(self) -> dict[str, Any]:
        """Get current GPS coordinates."""
        if self._gps is None:
            return {"lat": 0.0, "lon": 0.0, "gps_fix": False}

        reading = self._gps.reading
        return {
            "lat": reading.lat,
            "lon": reading.lon,
            "gps_fix": reading.has_fix
        }

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Perform environmental reading at current position.
        Called by grid driver or patrol executor action handler.
        """
        snapshot: EnvironmentSnapshot = self._take_snapshot()
        self._total_readings += 1
        self._store_snapshot(snapshot)
        self._publish_snapshot(snapshot)
        self._check_alerts(snapshot)

        return snapshot.to_dict()

    @property
    def latest_snapshot(self) -> Optional[EnvironmentSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def latest_trend(self) -> Optional[TrendAnalysis]:
        return self._latest_trend

    @property
    def total_readings(self) -> int:
        return self._total_readings

    @property
    def alerts_sent(self) -> int:
        return self._alerts_sent

    @property
    def recent_snapshots(self) -> list[EnvironmentSnapshot]:
        return list(self._snapshots[-20:])