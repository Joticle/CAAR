"""
Cerberus Microclimate Hotspot Mapper Head
Extendable arm + SHT45 high-precision probe + autonomous grid-pattern
driving + GPS-tagged data collection + self-generated heatmaps.
Maps temperature and humidity variations across the property to
identify hotspots, cold pockets, and moisture gradients.
"""

import time
import math
import logging
import statistics
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig
from cerberus.heads.base_head import BaseHead, HeadInfo


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MicroclimateReading:
    """A single georeferenced microclimate measurement."""
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    ground_temp_c: float = 0.0
    ambient_temp_c: float = 0.0
    temp_differential_c: float = 0.0
    heat_index_c: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    row: int = 0
    col: int = 0
    probe_height_cm: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": round(self.temperature_c, 2),
            "humidity_pct": round(self.humidity_pct, 2),
            "ground_temp_c": round(self.ground_temp_c, 2),
            "ambient_temp_c": round(self.ambient_temp_c, 2),
            "temp_differential_c": round(self.temp_differential_c, 2),
            "heat_index_c": round(self.heat_index_c, 1),
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "row": self.row,
            "col": self.col,
            "probe_height_cm": self.probe_height_cm,
            "timestamp": self.timestamp
        }


@dataclass
class HeatmapData:
    """Processed heatmap dataset for visualization."""
    name: str = ""
    rows: int = 0
    cols: int = 0
    temp_grid: list[list[float]] = field(default_factory=list)
    humidity_grid: list[list[float]] = field(default_factory=list)
    lat_grid: list[list[float]] = field(default_factory=list)
    lon_grid: list[list[float]] = field(default_factory=list)
    temp_min: float = 0.0
    temp_max: float = 0.0
    temp_avg: float = 0.0
    humidity_min: float = 0.0
    humidity_max: float = 0.0
    humidity_avg: float = 0.0
    hotspot_count: int = 0
    coldspot_count: int = 0
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "temp_grid": self.temp_grid,
            "humidity_grid": self.humidity_grid,
            "lat_grid": self.lat_grid,
            "lon_grid": self.lon_grid,
            "temp_min": round(self.temp_min, 1),
            "temp_max": round(self.temp_max, 1),
            "temp_avg": round(self.temp_avg, 1),
            "humidity_min": round(self.humidity_min, 1),
            "humidity_max": round(self.humidity_max, 1),
            "humidity_avg": round(self.humidity_avg, 1),
            "hotspot_count": self.hotspot_count,
            "coldspot_count": self.coldspot_count,
            "generated_at": self.generated_at
        }


@dataclass
class MicroclimateAnalysis:
    """Analysis summary of a completed microclimate survey."""
    survey_name: str = ""
    total_readings: int = 0
    duration_seconds: float = 0.0
    area_sq_m: float = 0.0
    temp_range_c: float = 0.0
    temp_avg: float = 0.0
    temp_stdev: float = 0.0
    humidity_range_pct: float = 0.0
    humidity_avg: float = 0.0
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    coldspots: list[dict[str, Any]] = field(default_factory=list)
    moisture_zones: list[dict[str, Any]] = field(default_factory=list)
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "survey_name": self.survey_name,
            "total_readings": self.total_readings,
            "duration_seconds": round(self.duration_seconds, 1),
            "area_sq_m": round(self.area_sq_m, 1),
            "temp_range_c": round(self.temp_range_c, 1),
            "temp_avg": round(self.temp_avg, 1),
            "temp_stdev": round(self.temp_stdev, 2),
            "humidity_range_pct": round(self.humidity_range_pct, 1),
            "humidity_avg": round(self.humidity_avg, 1),
            "hotspot_count": len(self.hotspots),
            "coldspot_count": len(self.coldspots),
            "moisture_zone_count": len(self.moisture_zones),
            "hotspots": self.hotspots,
            "coldspots": self.coldspots,
            "moisture_zones": self.moisture_zones,
            "generated_at": self.generated_at
        }


class MicroclimateHead(BaseHead):
    """
    Head 6: Microclimate Hotspot Mapper
    Deploys the SHT45 precision probe at each grid point to measure
    ground-level temperature and humidity. Builds georeferenced
    heatmaps that reveal microclimates across the property.

    Designed for grid driver integration — the grid driver moves
    Cerberus point by point, and at each stop this head:
        1. Extends probe arm to measurement height
        2. Waits for sensor stabilization
        3. Takes multiple readings and averages
        4. Retracts probe arm
        5. Logs georeferenced data point
        6. After grid complete → generates heatmap + analysis

    Las Vegas context: identifies heat islands (concrete/rock),
    shade pockets, moisture retention zones, and irrigation
    coverage gaps.
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="microclimate",
        description="Precision ground-level temperature and humidity mapping",
        version="1.0",
        requires_camera=False,
        requires_gps=True,
        requires_sensors=True,
        requires_audio=False,
        requires_servos=True,
        supported_tasks=["grid_scan", "scan", "station_keep"]
    )

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._probe_height_cm: float = self._config.get(
            "heads", "microclimate", "probe_height_cm", default=5.0
        )
        self._stabilization_seconds: float = self._config.get(
            "heads", "microclimate", "stabilization_seconds", default=3.0
        )
        self._samples_per_point: int = self._config.get(
            "heads", "microclimate", "samples_per_point", default=5
        )
        self._sample_interval: float = self._config.get(
            "heads", "microclimate", "sample_interval_seconds", default=1.0
        )
        self._hotspot_threshold_c: float = self._config.get(
            "heads", "microclimate", "hotspot_threshold_c", default=3.0
        )
        self._coldspot_threshold_c: float = self._config.get(
            "heads", "microclimate", "coldspot_threshold_c", default=-3.0
        )
        self._moisture_threshold_pct: float = self._config.get(
            "heads", "microclimate", "moisture_threshold_pct", default=15.0
        )
        self._save_path: str = self._config.get(
            "heads", "microclimate", "save_path", default="data/microclimate"
        )

        self._readings: list[MicroclimateReading] = []
        self._grid_rows: int = 0
        self._grid_cols: int = 0
        self._survey_start: float = 0.0
        self._survey_name: str = ""
        self._latest_heatmap: Optional[HeatmapData] = None
        self._latest_analysis: Optional[MicroclimateAnalysis] = None

        Path(self._save_path).mkdir(parents=True, exist_ok=True)

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def _on_load(self) -> bool:
        """Verify SHT45 sensor availability."""
        if self._environment is None:
            logger.warning("No environmental sensors bound — microclimate will simulate")

        logger.info(
            "Microclimate mapper configured — probe_height=%dcm, samples=%d, stabilization=%.1fs",
            int(self._probe_height_cm), self._samples_per_point, self._stabilization_seconds
        )
        return True

    def _on_start(self) -> bool:
        """Start microclimate survey."""
        self._readings = []
        self._survey_start = time.time()
        self._survey_name = f"survey_{time.strftime('%Y%m%d_%H%M%S')}"
        self._latest_heatmap = None
        self._latest_analysis = None

        logger.info("Microclimate survey started: %s", self._survey_name)
        return True

    def _on_stop(self) -> None:
        """Stop survey and generate outputs."""
        if self._readings:
            self._retract_probe()
            self._generate_heatmap()
            self._generate_analysis()

        logger.info(
            "Microclimate survey stopped: %s — %d readings",
            self._survey_name, len(self._readings)
        )

    def _on_unload(self) -> None:
        """Release microclimate resources."""
        logger.info("Microclimate mapper unloaded")

    def _run_cycle(self) -> None:
        """Continuous mode: take reading at current position."""
        reading: MicroclimateReading = self._take_reading()
        self._store_reading(reading)

        self._record_activity(
            f"Reading #{len(self._readings)}: "
            f"{reading.temperature_c:.1f}°C, "
            f"{reading.humidity_pct:.0f}% RH"
        )

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Take a precision microclimate measurement at a grid point.
        Called by grid driver action handler at each point.
        """
        row: int = getattr(point, "row", 0) if point else 0
        col: int = getattr(point, "col", 0) if point else 0

        self._grid_rows = max(self._grid_rows, row + 1)
        self._grid_cols = max(self._grid_cols, col + 1)

        self._extend_probe()
        self._wait_stabilization()

        reading: MicroclimateReading = self._take_reading(row=row, col=col)

        self._retract_probe()
        self._store_reading(reading)

        return reading.to_dict()

    def _take_reading(self, row: int = 0, col: int = 0) -> MicroclimateReading:
        """Take averaged microclimate reading from SHT45 probe."""
        temps: list[float] = []
        humids: list[float] = []

        for _ in range(self._samples_per_point):
            sample: dict[str, float] = self._read_probe()
            temps.append(sample["temperature_c"])
            humids.append(sample["humidity_pct"])
            time.sleep(self._sample_interval)

        avg_temp: float = statistics.mean(temps)
        avg_humid: float = statistics.mean(humids)

        ambient: dict[str, float] = self._read_ambient()
        ambient_temp: float = ambient.get("temperature_c", avg_temp)
        differential: float = avg_temp - ambient_temp

        heat_index: float = self._compute_heat_index(avg_temp, avg_humid)
        gps_data: dict[str, Any] = self._get_gps_data()

        return MicroclimateReading(
            temperature_c=avg_temp,
            humidity_pct=avg_humid,
            ground_temp_c=avg_temp,
            ambient_temp_c=ambient_temp,
            temp_differential_c=differential,
            heat_index_c=heat_index,
            lat=gps_data.get("lat", 0.0),
            lon=gps_data.get("lon", 0.0),
            row=row,
            col=col,
            probe_height_cm=self._probe_height_cm,
            timestamp=time.time()
        )

    def _read_probe(self) -> dict[str, float]:
        """Read the SHT45 probe sensor."""
        if self._environment is None:
            return self._simulated_probe_reading()

        try:
            reading = self._environment.reading
            return {
                "temperature_c": reading.probe_temp_c if hasattr(reading, "probe_temp_c") else reading.temperature_c,
                "humidity_pct": reading.probe_humidity_pct if hasattr(reading, "probe_humidity_pct") else reading.humidity_pct
            }
        except Exception as e:
            logger.error("SHT45 probe read error: %s", e)
            return {"temperature_c": 0.0, "humidity_pct": 0.0}

    def _read_ambient(self) -> dict[str, float]:
        """Read ambient temperature from BME680."""
        if self._environment is None:
            return self._simulated_ambient_reading()

        try:
            reading = self._environment.reading
            return {
                "temperature_c": reading.temperature_c,
                "humidity_pct": reading.humidity_pct
            }
        except Exception as e:
            logger.error("Ambient sensor read error: %s", e)
            return {"temperature_c": 0.0, "humidity_pct": 0.0}

    def _simulated_probe_reading(self) -> dict[str, float]:
        """Simulated SHT45 probe reading for dev environment."""
        import random
        base_temp: float = 38.0 + random.uniform(-5.0, 8.0)
        base_humid: float = 15.0 + random.uniform(-5.0, 10.0)
        return {
            "temperature_c": base_temp,
            "humidity_pct": max(5.0, base_humid)
        }

    def _simulated_ambient_reading(self) -> dict[str, float]:
        """Simulated ambient reading for dev environment."""
        import random
        return {
            "temperature_c": 36.0 + random.uniform(-2.0, 2.0),
            "humidity_pct": 18.0 + random.uniform(-3.0, 3.0)
        }

    def _extend_probe(self) -> None:
        """Extend the probe arm to measurement height."""
        logger.debug("Extending probe to %.0fcm", self._probe_height_cm)
        time.sleep(1.0)

    def _retract_probe(self) -> None:
        """Retract the probe arm to travel position."""
        logger.debug("Retracting probe")
        time.sleep(0.5)

    def _wait_stabilization(self) -> None:
        """Wait for sensor to stabilize after deployment."""
        logger.debug("Waiting %.1fs for sensor stabilization", self._stabilization_seconds)
        time.sleep(self._stabilization_seconds)

    @staticmethod
    def _compute_heat_index(temp_c: float, humidity: float) -> float:
        """Compute heat index using Rothfusz regression."""
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

    def _store_reading(self, reading: MicroclimateReading) -> None:
        """Store reading in memory and database."""
        self._readings.append(reading)

        if self._db is not None:
            try:
                self._db.log_sensor_reading(
                    sensor_type="microclimate",
                    data=reading.to_dict()
                )
            except Exception as e:
                logger.error("Failed to store microclimate reading: %s", e)

        if self._mqtt is not None:
            try:
                self._mqtt.publish_sensor_data("microclimate", reading.to_dict())
            except Exception:
                pass

    def _generate_heatmap(self) -> Optional[HeatmapData]:
        """Generate heatmap grids from collected readings."""
        if not self._readings:
            return None

        if self._grid_rows == 0 or self._grid_cols == 0:
            self._grid_rows = int(math.sqrt(len(self._readings)))
            self._grid_cols = max(1, len(self._readings) // max(1, self._grid_rows))

        temp_grid: list[list[float]] = [
            [0.0] * self._grid_cols for _ in range(self._grid_rows)
        ]
        humidity_grid: list[list[float]] = [
            [0.0] * self._grid_cols for _ in range(self._grid_rows)
        ]
        lat_grid: list[list[float]] = [
            [0.0] * self._grid_cols for _ in range(self._grid_rows)
        ]
        lon_grid: list[list[float]] = [
            [0.0] * self._grid_cols for _ in range(self._grid_rows)
        ]

        for reading in self._readings:
            r: int = min(reading.row, self._grid_rows - 1)
            c: int = min(reading.col, self._grid_cols - 1)
            temp_grid[r][c] = reading.temperature_c
            humidity_grid[r][c] = reading.humidity_pct
            lat_grid[r][c] = reading.lat
            lon_grid[r][c] = reading.lon

        all_temps: list[float] = [r.temperature_c for r in self._readings]
        all_humids: list[float] = [r.humidity_pct for r in self._readings]

        temp_avg: float = statistics.mean(all_temps)
        hotspot_count: int = sum(1 for t in all_temps if t > temp_avg + self._hotspot_threshold_c)
        coldspot_count: int = sum(1 for t in all_temps if t < temp_avg + self._coldspot_threshold_c)

        heatmap: HeatmapData = HeatmapData(
            name=self._survey_name,
            rows=self._grid_rows,
            cols=self._grid_cols,
            temp_grid=temp_grid,
            humidity_grid=humidity_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            temp_min=min(all_temps),
            temp_max=max(all_temps),
            temp_avg=temp_avg,
            humidity_min=min(all_humids),
            humidity_max=max(all_humids),
            humidity_avg=statistics.mean(all_humids),
            hotspot_count=hotspot_count,
            coldspot_count=coldspot_count,
            generated_at=time.time()
        )

        self._latest_heatmap = heatmap

        logger.info(
            "Heatmap generated: %dx%d grid, temp=%.1f-%.1f°C, %d hotspots, %d coldspots",
            self._grid_rows, self._grid_cols,
            heatmap.temp_min, heatmap.temp_max,
            hotspot_count, coldspot_count
        )

        if self._db is not None:
            try:
                self._db.log_sensor_reading(
                    sensor_type="microclimate_heatmap",
                    data=heatmap.to_dict()
                )
            except Exception as e:
                logger.error("Failed to store heatmap: %s", e)

        if self._mqtt is not None:
            try:
                self._mqtt.publish_sensor_data("microclimate_heatmap", heatmap.to_dict())
            except Exception:
                pass

        return heatmap

    def _generate_analysis(self) -> Optional[MicroclimateAnalysis]:
        """Generate survey analysis with hotspot/coldspot identification."""
        if not self._readings:
            return None

        all_temps: list[float] = [r.temperature_c for r in self._readings]
        all_humids: list[float] = [r.humidity_pct for r in self._readings]

        temp_avg: float = statistics.mean(all_temps)
        temp_stdev: float = statistics.stdev(all_temps) if len(all_temps) > 1 else 0.0
        humid_avg: float = statistics.mean(all_humids)

        hotspots: list[dict[str, Any]] = []
        coldspots: list[dict[str, Any]] = []
        moisture_zones: list[dict[str, Any]] = []

        for reading in self._readings:
            deviation: float = reading.temperature_c - temp_avg

            if deviation >= self._hotspot_threshold_c:
                hotspots.append({
                    "lat": round(reading.lat, 7),
                    "lon": round(reading.lon, 7),
                    "temp_c": round(reading.temperature_c, 1),
                    "deviation_c": round(deviation, 1),
                    "row": reading.row,
                    "col": reading.col
                })

            elif deviation <= self._coldspot_threshold_c:
                coldspots.append({
                    "lat": round(reading.lat, 7),
                    "lon": round(reading.lon, 7),
                    "temp_c": round(reading.temperature_c, 1),
                    "deviation_c": round(deviation, 1),
                    "row": reading.row,
                    "col": reading.col
                })

            humid_deviation: float = reading.humidity_pct - humid_avg
            if abs(humid_deviation) >= self._moisture_threshold_pct:
                moisture_zones.append({
                    "lat": round(reading.lat, 7),
                    "lon": round(reading.lon, 7),
                    "humidity_pct": round(reading.humidity_pct, 1),
                    "deviation_pct": round(humid_deviation, 1),
                    "type": "wet" if humid_deviation > 0 else "dry",
                    "row": reading.row,
                    "col": reading.col
                })

        duration: float = time.time() - self._survey_start if self._survey_start > 0 else 0.0

        analysis: MicroclimateAnalysis = MicroclimateAnalysis(
            survey_name=self._survey_name,
            total_readings=len(self._readings),
            duration_seconds=duration,
            temp_range_c=max(all_temps) - min(all_temps),
            temp_avg=temp_avg,
            temp_stdev=temp_stdev,
            humidity_range_pct=max(all_humids) - min(all_humids),
            humidity_avg=humid_avg,
            hotspots=hotspots,
            coldspots=coldspots,
            moisture_zones=moisture_zones,
            generated_at=time.time()
        )

        self._latest_analysis = analysis

        logger.info(
            "Survey analysis: %d readings, temp=%.1f±%.1f°C (range=%.1f°C), "
            "%d hotspots, %d coldspots, %d moisture zones",
            analysis.total_readings,
            analysis.temp_avg, analysis.temp_stdev, analysis.temp_range_c,
            len(hotspots), len(coldspots), len(moisture_zones)
        )

        if self._db is not None:
            try:
                self._db.log_sensor_reading(
                    sensor_type="microclimate_analysis",
                    data=analysis.to_dict()
                )
            except Exception as e:
                logger.error("Failed to store analysis: %s", e)

        return analysis

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

    @property
    def reading_count(self) -> int:
        return len(self._readings)

    @property
    def latest_heatmap(self) -> Optional[HeatmapData]:
        return self._latest_heatmap

    @property
    def latest_analysis(self) -> Optional[MicroclimateAnalysis]:
        return self._latest_analysis

    @property
    def recent_readings(self) -> list[MicroclimateReading]:
        return list(self._readings[-20:])

    @property
    def survey_name(self) -> str:
        return self._survey_name
