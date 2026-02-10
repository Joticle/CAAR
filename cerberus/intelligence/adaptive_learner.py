"""
Cerberus Adaptive Learner
On-device learning from historical data stored in SQLite. No GPU training,
no cloud — pure statistical analysis of data Cerberus already collects.
Makes the rover genuinely smarter over time.

Learning Modules:
    1. Confidence Calibration — adjusts detection thresholds based on
       verified prediction accuracy per model and label
    2. Patrol Optimization — analyzes detection history to focus patrol
       time on high-activity zones and reduce time in dead zones
    3. Weed Hotspot Prediction — builds a probability map of recurring
       weed locations, tightens grid spacing in hotspot areas
    4. Pest Activity Prediction — correlates pest appearances with time
       of day, temperature, and location to predict future activity
    5. Anomaly Detection — flags sensor readings that fall outside
       rolling statistical baselines, triggers investigation

All learning runs periodically (configurable interval) and writes
updated parameters back to the database and config. The rover's
behavior adapts without any manual tuning.
"""

import math
import time
import logging
import threading
from typing import Any, Optional
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of confidence calibration for a model."""
    model_name: str = ""
    total_verified: int = 0
    correct_count: int = 0
    accuracy_pct: float = 0.0
    avg_confidence: float = 0.0
    recommended_threshold: float = 0.5
    current_threshold: float = 0.5
    threshold_changed: bool = False
    label_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "verified": self.total_verified,
            "correct": self.correct_count,
            "accuracy_pct": round(self.accuracy_pct, 1),
            "avg_confidence": round(self.avg_confidence, 2),
            "recommended_threshold": round(self.recommended_threshold, 2),
            "current_threshold": round(self.current_threshold, 2),
            "threshold_changed": self.threshold_changed,
            "labels": self.label_stats
        }


@dataclass
class PatrolZone:
    """A zone within the patrol area with activity scoring."""
    lat_center: float = 0.0
    lon_center: float = 0.0
    detection_count: int = 0
    last_detection: str = ""
    activity_score: float = 0.0
    recommended_dwell_seconds: float = 10.0
    species_seen: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat_center, 6),
            "lon": round(self.lon_center, 6),
            "detections": self.detection_count,
            "activity_score": round(self.activity_score, 2),
            "dwell_seconds": round(self.recommended_dwell_seconds, 1),
            "species": self.species_seen
        }


@dataclass
class WeedHotspot:
    """A location with recurring weed detections."""
    lat: float = 0.0
    lon: float = 0.0
    detection_count: int = 0
    avg_confidence: float = 0.0
    species_found: list[str] = field(default_factory=list)
    recurrence_probability: float = 0.0
    recommended_grid_spacing_m: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "detections": self.detection_count,
            "avg_confidence": round(self.avg_confidence, 2),
            "species": self.species_found,
            "recurrence_prob": round(self.recurrence_probability, 2),
            "grid_spacing_m": round(self.recommended_grid_spacing_m, 2)
        }


@dataclass
class PestPrediction:
    """Predicted pest activity for a time window and location."""
    species: str = ""
    hour_of_day: int = 0
    probability: float = 0.0
    avg_temp_c: float = 0.0
    hot_lat: float = 0.0
    hot_lon: float = 0.0
    total_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "hour": self.hour_of_day,
            "probability": round(self.probability, 2),
            "avg_temp_c": round(self.avg_temp_c, 1),
            "lat": round(self.hot_lat, 6),
            "lon": round(self.hot_lon, 6),
            "events": self.total_events
        }


@dataclass
class AnomalyAlert:
    """A sensor reading that deviates significantly from baseline."""
    sensor_type: str = ""
    metric: str = ""
    value: float = 0.0
    baseline_mean: float = 0.0
    baseline_stdev: float = 0.0
    deviation_sigma: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor_type,
            "metric": self.metric,
            "value": round(self.value, 2),
            "mean": round(self.baseline_mean, 2),
            "stdev": round(self.baseline_stdev, 2),
            "sigma": round(self.deviation_sigma, 1),
            "timestamp": self.timestamp
        }


class AdaptiveLearner:
    """
    Runs periodic learning passes over historical data in SQLite.
    Updates detection thresholds, patrol priorities, hotspot maps,
    pest predictions, and anomaly baselines. All computation is
    lightweight SQL aggregation — no ML frameworks required.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._db: Optional[Any] = None

        self._learning_interval_s: float = config.get("learning", "interval_seconds", default=3600.0)
        self._min_samples_calibration: int = config.get("learning", "min_samples_calibration", default=20)
        self._min_samples_patrol: int = config.get("learning", "min_samples_patrol", default=10)
        self._anomaly_sigma_threshold: float = config.get("learning", "anomaly_sigma_threshold", default=2.0)
        self._baseline_hours: int = config.get("learning", "baseline_hours", default=168)
        self._default_dwell_s: float = config.get("mission", "patrol_dwell_seconds", default=10.0)
        self._default_grid_spacing_m: float = config.get("mission", "grid_spacing_m", default=2.0)
        self._zone_radius_deg: float = config.get("learning", "zone_radius_deg", default=0.00005)

        self._calibration_results: dict[str, CalibrationResult] = {}
        self._patrol_zones: list[PatrolZone] = []
        self._weed_hotspots: list[WeedHotspot] = []
        self._pest_predictions: list[PestPrediction] = []
        self._anomalies: list[AnomalyAlert] = []

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._last_learning_run: float = 0.0
        self._total_runs: int = 0

        logger.info(
            "Adaptive learner initialized — interval=%ds, anomaly_sigma=%.1f, baseline=%dh",
            int(self._learning_interval_s), self._anomaly_sigma_threshold, self._baseline_hours
        )

    def bind_db(self, db: Any) -> None:
        """Bind the database for data access."""
        self._db = db

    def start(self) -> bool:
        """Start the background learning loop."""
        if self._db is None:
            logger.error("Cannot start learner — no database bound")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._learning_loop,
            name="adaptive-learner",
            daemon=True
        )
        self._thread.start()
        logger.info("Adaptive learner started")
        return True

    def stop(self) -> None:
        """Stop the background learning loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Adaptive learner stopped — %d total runs", self._total_runs)

    def _learning_loop(self) -> None:
        """Background thread that runs learning passes periodically."""
        while self._running:
            now: float = time.time()
            elapsed: float = now - self._last_learning_run

            if elapsed >= self._learning_interval_s:
                try:
                    self.run_all()
                except Exception as e:
                    logger.error("Learning run failed: %s", e)
                self._last_learning_run = time.time()

            time.sleep(30.0)

    def run_all(self) -> dict[str, Any]:
        """Execute all learning modules. Returns summary of results."""
        start: float = time.time()
        self._total_runs += 1

        logger.info("Learning run #%d starting", self._total_runs)

        results: dict[str, Any] = {
            "run_number": self._total_runs,
            "calibration": {},
            "patrol_zones": 0,
            "weed_hotspots": 0,
            "pest_predictions": 0,
            "anomalies": 0
        }

        try:
            self._calibration_results = self._run_confidence_calibration()
            results["calibration"] = {
                name: r.to_dict() for name, r in self._calibration_results.items()
            }
        except Exception as e:
            logger.error("Confidence calibration failed: %s", e)

        try:
            self._patrol_zones = self._run_patrol_optimization()
            results["patrol_zones"] = len(self._patrol_zones)
        except Exception as e:
            logger.error("Patrol optimization failed: %s", e)

        try:
            self._weed_hotspots = self._run_weed_hotspot_analysis()
            results["weed_hotspots"] = len(self._weed_hotspots)
        except Exception as e:
            logger.error("Weed hotspot analysis failed: %s", e)

        try:
            self._pest_predictions = self._run_pest_prediction()
            results["pest_predictions"] = len(self._pest_predictions)
        except Exception as e:
            logger.error("Pest prediction failed: %s", e)

        try:
            self._anomalies = self._run_anomaly_detection()
            results["anomalies"] = len(self._anomalies)
        except Exception as e:
            logger.error("Anomaly detection failed: %s", e)

        elapsed_ms: float = (time.time() - start) * 1000
        results["duration_ms"] = round(elapsed_ms, 1)

        logger.info(
            "Learning run #%d complete in %.0fms — calibrated %d models, %d zones, "
            "%d hotspots, %d predictions, %d anomalies",
            self._total_runs, elapsed_ms,
            len(self._calibration_results),
            len(self._patrol_zones),
            len(self._weed_hotspots),
            len(self._pest_predictions),
            len(self._anomalies)
        )

        return results

    # =================================================================
    # MODULE 1: CONFIDENCE CALIBRATION
    # =================================================================

    def _run_confidence_calibration(self) -> dict[str, CalibrationResult]:
        """
        Analyze verified predictions to determine if detection thresholds
        should be adjusted. If the weed detector triggers at 60% confidence
        but only 40% of those are confirmed weeds, raise the threshold.
        """
        results: dict[str, CalibrationResult] = {}

        models: list[str] = [
            "weed_detector", "threat_classifier",
            "wildlife_classifier", "bird_classifier"
        ]

        for model_name in models:
            model_stats: Optional[dict[str, Any]] = self._db.get_model_accuracy(model_name)

            if model_stats is None or (model_stats.get("total_verified") or 0) < self._min_samples_calibration:
                continue

            total: int = model_stats["total_verified"]
            correct: int = model_stats.get("correct_count") or 0
            accuracy: float = (correct / total * 100) if total > 0 else 0.0
            avg_conf: float = model_stats.get("avg_confidence") or 0.5

            current_threshold: float = self._get_model_threshold(model_name)
            recommended: float = self._calculate_recommended_threshold(
                accuracy, avg_conf, current_threshold
            )

            label_stats: dict[str, dict[str, Any]] = self._get_label_breakdown(model_name)

            result: CalibrationResult = CalibrationResult(
                model_name=model_name,
                total_verified=total,
                correct_count=correct,
                accuracy_pct=accuracy,
                avg_confidence=avg_conf,
                recommended_threshold=recommended,
                current_threshold=current_threshold,
                threshold_changed=abs(recommended - current_threshold) >= 0.05,
                label_stats=label_stats
            )

            results[model_name] = result

            if result.threshold_changed:
                logger.info(
                    "Calibration: %s threshold %.2f -> %.2f (accuracy %.1f%% on %d samples)",
                    model_name, current_threshold, recommended, accuracy, total
                )

        return results

    def _calculate_recommended_threshold(
        self, accuracy_pct: float, avg_confidence: float, current: float
    ) -> float:
        """
        Calculate a recommended confidence threshold based on accuracy.
        High accuracy -> can lower threshold (catch more)
        Low accuracy -> raise threshold (reduce false positives)
        """
        if accuracy_pct >= 90.0:
            adjustment: float = -0.05
        elif accuracy_pct >= 75.0:
            adjustment = 0.0
        elif accuracy_pct >= 60.0:
            adjustment = 0.05
        elif accuracy_pct >= 40.0:
            adjustment = 0.10
        else:
            adjustment = 0.15

        recommended: float = current + adjustment
        recommended = max(0.3, min(0.95, recommended))

        return round(recommended, 2)

    def _get_model_threshold(self, model_name: str) -> float:
        """Get the current threshold for a model from config."""
        head_map: dict[str, str] = {
            "weed_detector": "weed_scanner",
            "threat_classifier": "surveillance",
            "wildlife_classifier": "pest_deterrent",
            "bird_classifier": "bird_watcher"
        }
        head_name: str = head_map.get(model_name, "")
        if head_name:
            return self._config.get("heads", head_name, "confidence_threshold", default=0.5)
        return 0.5

    def _get_label_breakdown(self, model_name: str) -> dict[str, dict[str, Any]]:
        """Get per-label accuracy breakdown for a model."""
        label_stats: dict[str, dict[str, Any]] = {}

        try:
            rows: list[dict[str, Any]] = self._db.query(
                "SELECT label, COUNT(*) as total, "
                "SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct, "
                "AVG(predicted_confidence) as avg_conf "
                "FROM confidence_tracking WHERE model_name = ? AND verified = 1 "
                "GROUP BY label",
                (model_name,)
            )

            for row in rows:
                label: str = row["label"]
                total: int = row["total"]
                correct: int = row.get("correct") or 0
                label_stats[label] = {
                    "total": total,
                    "correct": correct,
                    "accuracy_pct": round(correct / total * 100, 1) if total > 0 else 0.0,
                    "avg_confidence": round(row.get("avg_conf") or 0.0, 2)
                }
        except Exception as e:
            logger.error("Label breakdown failed for %s: %s", model_name, e)

        return label_stats

    # =================================================================
    # MODULE 2: PATROL OPTIMIZATION
    # =================================================================

    def _run_patrol_optimization(self) -> list[PatrolZone]:
        """
        Analyze detection history to score patrol zones. High-activity
        zones get longer dwell times, low-activity zones get shorter.
        Results feed into the patrol executor's dwell time per waypoint.
        """
        zones: list[PatrolZone] = []

        try:
            rows: list[dict[str, Any]] = self._db.query(
                "SELECT ROUND(gps_lat, 5) as lat_zone, ROUND(gps_lon, 5) as lon_zone, "
                "COUNT(*) as detection_count, MAX(timestamp) as last_detection, "
                "GROUP_CONCAT(DISTINCT label) as labels "
                "FROM detections WHERE timestamp > datetime('now', ?) "
                "AND gps_lat IS NOT NULL AND gps_lat != 0 "
                "GROUP BY lat_zone, lon_zone ORDER BY detection_count DESC",
                (f"-{self._baseline_hours} hours",)
            )

            if not rows:
                return zones

            max_count: int = max(r["detection_count"] for r in rows)

            for row in rows:
                if row["detection_count"] < self._min_samples_patrol:
                    continue

                score: float = row["detection_count"] / max_count if max_count > 0 else 0.0

                min_dwell: float = self._default_dwell_s * 0.5
                max_dwell: float = self._default_dwell_s * 3.0
                recommended_dwell: float = min_dwell + score * (max_dwell - min_dwell)

                labels: list[str] = []
                if row.get("labels"):
                    labels = [l.strip() for l in row["labels"].split(",") if l.strip()]

                zones.append(PatrolZone(
                    lat_center=row["lat_zone"],
                    lon_center=row["lon_zone"],
                    detection_count=row["detection_count"],
                    last_detection=row.get("last_detection", ""),
                    activity_score=score,
                    recommended_dwell_seconds=recommended_dwell,
                    species_seen=labels
                ))

        except Exception as e:
            logger.error("Patrol optimization query failed: %s", e)

        logger.info("Patrol optimization: %d active zones identified", len(zones))
        return zones

    def get_zone_dwell_adjustment(self, lat: float, lon: float) -> float:
        """
        Get the recommended dwell time for a GPS position based on
        historical activity. Returns the default if no zone data exists.
        """
        for zone in self._patrol_zones:
            dlat: float = abs(lat - zone.lat_center)
            dlon: float = abs(lon - zone.lon_center)
            if dlat <= self._zone_radius_deg and dlon <= self._zone_radius_deg:
                return zone.recommended_dwell_seconds

        return self._default_dwell_s

    # =================================================================
    # MODULE 3: WEED HOTSPOT PREDICTION
    # =================================================================

    def _run_weed_hotspot_analysis(self) -> list[WeedHotspot]:
        """
        Build a probability map of recurring weed locations.
        Locations with repeated detections get tighter grid spacing
        in future scans.
        """
        hotspots: list[WeedHotspot] = []

        try:
            rows: list[dict[str, Any]] = self._db.get_weed_hotspots(min_detections=2)

            if not rows:
                return hotspots

            max_count: int = max(r["detection_count"] for r in rows) if rows else 1

            for row in rows:
                count: int = row["detection_count"]
                probability: float = min(1.0, count / (max_count * 1.2))

                if probability >= 0.7:
                    spacing: float = self._default_grid_spacing_m * 0.5
                elif probability >= 0.4:
                    spacing = self._default_grid_spacing_m * 0.75
                else:
                    spacing = self._default_grid_spacing_m

                species: list[str] = []
                if row.get("species_found"):
                    species = [s.strip() for s in row["species_found"].split(",") if s.strip()]

                hotspots.append(WeedHotspot(
                    lat=row["lat_zone"],
                    lon=row["lon_zone"],
                    detection_count=count,
                    avg_confidence=row.get("avg_confidence", 0.0),
                    species_found=species,
                    recurrence_probability=probability,
                    recommended_grid_spacing_m=spacing
                ))

        except Exception as e:
            logger.error("Weed hotspot analysis failed: %s", e)

        logger.info("Weed analysis: %d hotspots identified", len(hotspots))
        return hotspots

    def get_grid_spacing_for_location(self, lat: float, lon: float) -> float:
        """
        Get recommended grid spacing for a location based on weed history.
        Tighter spacing in hotspot areas means more thorough scanning.
        """
        for hotspot in self._weed_hotspots:
            dlat: float = abs(lat - hotspot.lat)
            dlon: float = abs(lon - hotspot.lon)
            if dlat <= self._zone_radius_deg and dlon <= self._zone_radius_deg:
                return hotspot.recommended_grid_spacing_m

        return self._default_grid_spacing_m

    # =================================================================
    # MODULE 4: PEST ACTIVITY PREDICTION
    # =================================================================

    def _run_pest_prediction(self) -> list[PestPrediction]:
        """
        Correlate pest appearances with time of day, temperature, and
        location. Predict when and where pests are most likely to appear
        so Cerberus can pre-position at hotspots.
        """
        predictions: list[PestPrediction] = []

        try:
            rows: list[dict[str, Any]] = self._db.query(
                "SELECT pe.species, "
                "CAST(strftime('%H', pe.timestamp) AS INTEGER) as hour, "
                "COUNT(*) as event_count, "
                "AVG(pe.gps_lat) as avg_lat, AVG(pe.gps_lon) as avg_lon "
                "FROM pest_events pe "
                "WHERE pe.timestamp > datetime('now', ?) "
                "AND pe.gps_lat IS NOT NULL AND pe.gps_lat != 0 "
                "GROUP BY pe.species, hour "
                "ORDER BY event_count DESC",
                (f"-{self._baseline_hours} hours",)
            )

            if not rows:
                return predictions

            total_events: int = sum(r["event_count"] for r in rows)

            temp_rows: list[dict[str, Any]] = self._db.query(
                "SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, "
                "AVG(temperature_c) as avg_temp "
                "FROM sensor_readings WHERE sensor_type = 'bme680' "
                "AND timestamp > datetime('now', ?) "
                "GROUP BY hour",
                (f"-{self._baseline_hours} hours",)
            )
            temp_by_hour: dict[int, float] = {
                r["hour"]: r["avg_temp"] for r in temp_rows if r.get("avg_temp") is not None
            }

            for row in rows:
                probability: float = row["event_count"] / total_events if total_events > 0 else 0.0
                hour: int = row["hour"]

                predictions.append(PestPrediction(
                    species=row["species"],
                    hour_of_day=hour,
                    probability=probability,
                    avg_temp_c=temp_by_hour.get(hour, 0.0),
                    hot_lat=row.get("avg_lat", 0.0),
                    hot_lon=row.get("avg_lon", 0.0),
                    total_events=row["event_count"]
                ))

        except Exception as e:
            logger.error("Pest prediction failed: %s", e)

        logger.info("Pest prediction: %d species/hour patterns identified", len(predictions))
        return predictions

    def get_pest_risk_now(self) -> list[PestPrediction]:
        """Get pest predictions relevant to the current hour."""
        current_hour: int = int(time.strftime("%H"))
        window: set[int] = {(current_hour - 1) % 24, current_hour, (current_hour + 1) % 24}
        return [p for p in self._pest_predictions if p.hour_of_day in window]

    # =================================================================
    # MODULE 5: ANOMALY DETECTION
    # =================================================================

    def _run_anomaly_detection(self) -> list[AnomalyAlert]:
        """
        Flag sensor readings that deviate significantly from rolling
        baselines. Uses simple sigma-based detection on the last
        baseline_hours of data.
        """
        anomalies: list[AnomalyAlert] = []

        sensor_metrics: list[tuple[str, str]] = [
            ("bme680", "temperature_c"),
            ("bme680", "humidity_pct"),
            ("bme680", "pressure_hpa"),
            ("bme680", "gas_resistance_ohms"),
            ("scd40", "co2_ppm"),
            ("sht45", "temperature_c"),
            ("sht45", "humidity_pct"),
        ]

        for sensor_type, metric in sensor_metrics:
            try:
                alerts: list[AnomalyAlert] = self._check_metric_anomalies(sensor_type, metric)
                anomalies.extend(alerts)
            except Exception as e:
                logger.error("Anomaly check failed for %s.%s: %s", sensor_type, metric, e)

        if anomalies:
            logger.warning("Anomaly detection: %d anomalies found", len(anomalies))

        return anomalies

    def _check_metric_anomalies(self, sensor_type: str, metric: str) -> list[AnomalyAlert]:
        """Check a single sensor metric for anomalies against its rolling baseline."""
        alerts: list[AnomalyAlert] = []

        baseline: Optional[dict[str, Any]] = self._db.query_one(
            f"SELECT AVG({metric}) as mean, "
            f"AVG({metric} * {metric}) - AVG({metric}) * AVG({metric}) as variance, "
            f"COUNT(*) as sample_count "
            f"FROM sensor_readings WHERE sensor_type = ? "
            f"AND {metric} IS NOT NULL "
            f"AND timestamp > datetime('now', ?)",
            (sensor_type, f"-{self._baseline_hours} hours")
        )

        if baseline is None or (baseline.get("sample_count") or 0) < 10:
            return alerts

        mean: float = baseline.get("mean") or 0.0
        variance: float = baseline.get("variance") or 0.0
        if variance < 0:
            variance = 0.0
        stdev: float = math.sqrt(variance)

        if stdev < 0.001:
            return alerts

        recent: list[dict[str, Any]] = self._db.query(
            f"SELECT {metric} as value, timestamp "
            f"FROM sensor_readings WHERE sensor_type = ? "
            f"AND {metric} IS NOT NULL "
            f"AND timestamp > datetime('now', '-1 hours') "
            f"ORDER BY timestamp DESC",
            (sensor_type,),
            limit=10
        )

        for row in recent:
            value: float = row.get("value") or 0.0
            deviation: float = abs(value - mean) / stdev

            if deviation >= self._anomaly_sigma_threshold:
                alerts.append(AnomalyAlert(
                    sensor_type=sensor_type,
                    metric=metric,
                    value=value,
                    baseline_mean=mean,
                    baseline_stdev=stdev,
                    deviation_sigma=deviation,
                    timestamp=row.get("timestamp", "")
                ))

        return alerts

    def get_active_anomalies(self) -> list[AnomalyAlert]:
        """Get anomalies from the most recent learning run."""
        return list(self._anomalies)

    # =================================================================
    # STATE AND PROPERTIES
    # =================================================================

    @property
    def calibration_results(self) -> dict[str, CalibrationResult]:
        return dict(self._calibration_results)

    @property
    def patrol_zones(self) -> list[PatrolZone]:
        return list(self._patrol_zones)

    @property
    def weed_hotspots(self) -> list[WeedHotspot]:
        return list(self._weed_hotspots)

    @property
    def pest_predictions(self) -> list[PestPrediction]:
        return list(self._pest_predictions)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def total_runs(self) -> int:
        return self._total_runs

    def stats(self) -> dict[str, Any]:
        return {
            "total_runs": self._total_runs,
            "running": self._running,
            "calibrated_models": len(self._calibration_results),
            "patrol_zones": len(self._patrol_zones),
            "weed_hotspots": len(self._weed_hotspots),
            "pest_predictions": len(self._pest_predictions),
            "active_anomalies": len(self._anomalies),
            "learning_interval_s": self._learning_interval_s,
            "baseline_hours": self._baseline_hours
        }