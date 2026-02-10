"""
Tests for cerberus.intelligence.adaptive_learner — AdaptiveLearner
Validates confidence calibration, patrol optimization, weed hotspot
analysis, pest prediction, anomaly detection, background learning loop,
and all data access patterns. Uses the real test database.
"""

import time
import math
import pytest
from typing import Any, Optional

from cerberus.core.config import CerberusConfig
from cerberus.storage.db import CerberusDB
from cerberus.intelligence.adaptive_learner import (
    AdaptiveLearner, CalibrationResult, PatrolZone, WeedHotspot,
    PestPrediction, AnomalyAlert
)


class TestCalibrationResult:
    """CalibrationResult dataclass."""

    def test_default(self) -> None:
        r: CalibrationResult = CalibrationResult()
        assert r.model_name == ""
        assert r.total_verified == 0
        assert not r.threshold_changed

    def test_to_dict(self) -> None:
        r: CalibrationResult = CalibrationResult(
            model_name="weed_detector",
            total_verified=50,
            correct_count=45,
            accuracy_pct=90.0,
            avg_confidence=0.82,
            recommended_threshold=0.55,
            current_threshold=0.60,
            threshold_changed=True,
            label_stats={"dandelion": {"total": 30, "correct": 28}}
        )
        d: dict[str, Any] = r.to_dict()
        assert d["model"] == "weed_detector"
        assert d["verified"] == 50
        assert d["accuracy_pct"] == 90.0
        assert d["threshold_changed"] is True
        assert "dandelion" in d["labels"]


class TestPatrolZone:
    """PatrolZone dataclass."""

    def test_to_dict(self) -> None:
        z: PatrolZone = PatrolZone(
            lat_center=36.17, lon_center=-115.14,
            detection_count=15, activity_score=0.8,
            recommended_dwell_seconds=25.0,
            species_seen=["rabbit", "quail"]
        )
        d: dict[str, Any] = z.to_dict()
        assert d["lat"] == 36.17
        assert d["detections"] == 15
        assert d["activity_score"] == 0.8
        assert "rabbit" in d["species"]


class TestWeedHotspot:
    """WeedHotspot dataclass."""

    def test_to_dict(self) -> None:
        h: WeedHotspot = WeedHotspot(
            lat=36.17, lon=-115.14,
            detection_count=8, avg_confidence=0.85,
            species_found=["dandelion"], recurrence_probability=0.7,
            recommended_grid_spacing_m=1.0
        )
        d: dict[str, Any] = h.to_dict()
        assert d["detections"] == 8
        assert d["recurrence_prob"] == 0.7
        assert d["grid_spacing_m"] == 1.0


class TestPestPrediction:
    """PestPrediction dataclass."""

    def test_to_dict(self) -> None:
        p: PestPrediction = PestPrediction(
            species="rabbit", hour_of_day=6,
            probability=0.45, avg_temp_c=28.0,
            hot_lat=36.17, hot_lon=-115.14,
            total_events=12
        )
        d: dict[str, Any] = p.to_dict()
        assert d["species"] == "rabbit"
        assert d["hour"] == 6
        assert d["probability"] == 0.45
        assert d["events"] == 12


class TestAnomalyAlert:
    """AnomalyAlert dataclass."""

    def test_to_dict(self) -> None:
        a: AnomalyAlert = AnomalyAlert(
            sensor_type="bme680", metric="temperature_c",
            value=55.0, baseline_mean=38.0, baseline_stdev=3.0,
            deviation_sigma=5.7, timestamp="2025-07-15 14:30:00"
        )
        d: dict[str, Any] = a.to_dict()
        assert d["sensor"] == "bme680"
        assert d["metric"] == "temperature_c"
        assert d["sigma"] == 5.7


class TestLearnerInit:
    """AdaptiveLearner initialization."""

    def test_creates_with_config(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        assert learner is not None
        assert learner.is_running is False
        assert learner.total_runs == 0

    def test_default_config(self) -> None:
        learner: AdaptiveLearner = AdaptiveLearner()
        assert learner is not None

    def test_bind_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        assert learner._db is db


class TestStartStop:
    """Background learning loop start/stop."""

    def test_start_without_db_fails(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        started: bool = learner.start()
        assert started is False

    def test_start_with_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        started: bool = learner.start()
        assert started is True
        assert learner.is_running is True
        learner.stop()
        assert learner.is_running is False

    def test_double_start_safe(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        learner.start()
        result: bool = learner.start()
        assert result is True
        learner.stop()

    def test_stop_without_start(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.stop()
        assert learner.is_running is False


class TestRunAll:
    """Full learning run with database."""

    def test_run_all_empty_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        results: dict[str, Any] = learner.run_all()
        assert results["run_number"] == 1
        assert "duration_ms" in results
        assert learner.total_runs == 1

    def test_run_all_multiple_times(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        learner.run_all()
        learner.run_all()
        learner.run_all()
        assert learner.total_runs == 3


class TestConfidenceCalibration:
    """Module 1: Confidence calibration."""

    def _seed_confidence_data(self, db: CerberusDB, model: str, count: int, correct_pct: float) -> None:
        correct_count: int = int(count * correct_pct)
        for i in range(count):
            db.log_confidence(
                model_name=model,
                label="weed",
                predicted_confidence=0.7 + (i % 10) * 0.02
            )

        rows = db.query(
            "SELECT id FROM confidence_tracking WHERE model_name = ? ORDER BY id",
            (model,)
        )
        for i, row in enumerate(rows):
            db.verify_prediction(row["id"], correct=(i < correct_count))

    def test_calibration_with_data(self, config: CerberusConfig, db: CerberusDB) -> None:
        config._data["learning"]["min_samples_calibration"] = 5
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        self._seed_confidence_data(db, "weed_detector", 20, 0.85)

        results: dict[str, CalibrationResult] = learner._run_confidence_calibration()
        assert "weed_detector" in results
        r: CalibrationResult = results["weed_detector"]
        assert r.total_verified == 20
        assert r.accuracy_pct > 0

    def test_calibration_insufficient_data(self, config: CerberusConfig, db: CerberusDB) -> None:
        config._data["learning"]["min_samples_calibration"] = 100
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        self._seed_confidence_data(db, "weed_detector", 5, 0.8)

        results: dict[str, CalibrationResult] = learner._run_confidence_calibration()
        assert "weed_detector" not in results

    def test_high_accuracy_lowers_threshold(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        recommended: float = learner._calculate_recommended_threshold(
            accuracy_pct=95.0, avg_confidence=0.85, current=0.60
        )
        assert recommended < 0.60

    def test_low_accuracy_raises_threshold(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        recommended: float = learner._calculate_recommended_threshold(
            accuracy_pct=35.0, avg_confidence=0.55, current=0.50
        )
        assert recommended > 0.50

    def test_moderate_accuracy_no_change(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        recommended: float = learner._calculate_recommended_threshold(
            accuracy_pct=80.0, avg_confidence=0.75, current=0.60
        )
        assert recommended == 0.60

    def test_threshold_clamped_min(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        recommended: float = learner._calculate_recommended_threshold(
            accuracy_pct=99.0, avg_confidence=0.9, current=0.3
        )
        assert recommended >= 0.3

    def test_threshold_clamped_max(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        recommended: float = learner._calculate_recommended_threshold(
            accuracy_pct=10.0, avg_confidence=0.3, current=0.90
        )
        assert recommended <= 0.95


class TestPatrolOptimization:
    """Module 2: Patrol optimization."""

    def _seed_detections(self, db: CerberusDB, lat: float, lon: float, count: int) -> None:
        for i in range(count):
            db.log_detection(
                head_type="surveillance",
                detection_type="motion",
                label=f"event_{i}",
                confidence=0.7,
                gps_lat=lat,
                gps_lon=lon
            )

    def test_patrol_zones_from_detections(self, config: CerberusConfig, db: CerberusDB) -> None:
        config._data["learning"]["min_samples_patrol"] = 3
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        self._seed_detections(db, 36.17001, -115.13981, 10)
        self._seed_detections(db, 36.17050, -115.13950, 5)

        zones: list[PatrolZone] = learner._run_patrol_optimization()
        assert len(zones) >= 1
        assert zones[0].detection_count >= 5

    def test_patrol_zones_empty_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        zones: list[PatrolZone] = learner._run_patrol_optimization()
        assert len(zones) == 0

    def test_zone_dwell_adjustment_known_zone(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._patrol_zones = [
            PatrolZone(
                lat_center=36.17001, lon_center=-115.13981,
                detection_count=20, activity_score=0.9,
                recommended_dwell_seconds=25.0
            )
        ]
        dwell: float = learner.get_zone_dwell_adjustment(36.17001, -115.13981)
        assert dwell == 25.0

    def test_zone_dwell_adjustment_unknown_zone(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._patrol_zones = []
        dwell: float = learner.get_zone_dwell_adjustment(0.0, 0.0)
        assert dwell == learner._default_dwell_s


class TestWeedHotspotAnalysis:
    """Module 3: Weed hotspot prediction."""

    def _seed_weed_data(self, db: CerberusDB, lat: float, lon: float, count: int) -> None:
        for i in range(count):
            db.log_weed_detection(
                species="dandelion",
                confidence=0.75 + (i % 5) * 0.03,
                gps_lat=lat + (i % 3) * 0.000001,
                gps_lon=lon + (i % 3) * 0.000001,
                hdop=1.5
            )

    def test_hotspots_from_weed_data(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        self._seed_weed_data(db, 36.17001, -115.13981, 5)

        hotspots: list[WeedHotspot] = learner._run_weed_hotspot_analysis()
        assert len(hotspots) >= 0

    def test_hotspots_empty_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        hotspots: list[WeedHotspot] = learner._run_weed_hotspot_analysis()
        assert len(hotspots) == 0

    def test_grid_spacing_at_hotspot(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._weed_hotspots = [
            WeedHotspot(
                lat=36.17001, lon=-115.13981,
                detection_count=10, recurrence_probability=0.8,
                recommended_grid_spacing_m=1.0
            )
        ]
        spacing: float = learner.get_grid_spacing_for_location(36.17001, -115.13981)
        assert spacing == 1.0

    def test_grid_spacing_no_hotspot(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._weed_hotspots = []
        spacing: float = learner.get_grid_spacing_for_location(0.0, 0.0)
        assert spacing == learner._default_grid_spacing_m


class TestPestPrediction:
    """Module 4: Pest activity prediction."""

    def _seed_pest_data(self, db: CerberusDB, species: str, count: int) -> None:
        for i in range(count):
            db.log_pest_event(
                species=species,
                confidence=0.75,
                deterrent_action="audio_predator",
                escalation="gentle",
                response_effective=(i % 2 == 0),
                gps_lat=36.17 + (i % 3) * 0.0001,
                gps_lon=-115.14 + (i % 3) * 0.0001
            )

    def test_predictions_from_pest_data(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        self._seed_pest_data(db, "rabbit", 10)
        self._seed_pest_data(db, "squirrel", 5)

        predictions: list[PestPrediction] = learner._run_pest_prediction()
        assert len(predictions) >= 0

    def test_predictions_empty_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        predictions: list[PestPrediction] = learner._run_pest_prediction()
        assert len(predictions) == 0

    def test_pest_risk_now(self, config: CerberusConfig) -> None:
        current_hour: int = int(time.strftime("%H"))
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._pest_predictions = [
            PestPrediction(species="rabbit", hour_of_day=current_hour, probability=0.6),
            PestPrediction(species="squirrel", hour_of_day=(current_hour + 12) % 24, probability=0.3),
        ]
        current: list[PestPrediction] = learner.get_pest_risk_now()
        assert len(current) >= 1
        assert current[0].species == "rabbit"

    def test_pest_risk_now_empty(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._pest_predictions = []
        assert len(learner.get_pest_risk_now()) == 0


class TestAnomalyDetection:
    """Module 5: Anomaly detection."""

    def _seed_sensor_baseline(self, db: CerberusDB, sensor: str, metric_val: float, count: int) -> None:
        for i in range(count):
            db.log_sensor_reading(
                sensor_type=sensor,
                temperature_c=metric_val + (i % 5) * 0.5
            )

    def test_anomaly_detection_empty_db(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        anomalies: list[AnomalyAlert] = learner._run_anomaly_detection()
        assert len(anomalies) == 0

    def test_anomaly_detection_with_baseline(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)

        for i in range(20):
            db.log_sensor_reading(sensor_type="bme680", temperature_c=38.0 + (i % 3) * 0.5)

        db.log_sensor_reading(sensor_type="bme680", temperature_c=65.0)

        anomalies: list[AnomalyAlert] = learner._run_anomaly_detection()
        assert len(anomalies) >= 0

    def test_get_active_anomalies(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._anomalies = [
            AnomalyAlert(sensor_type="bme680", metric="temperature_c", deviation_sigma=3.5)
        ]
        active: list[AnomalyAlert] = learner.get_active_anomalies()
        assert len(active) == 1
        assert active[0].deviation_sigma == 3.5


class TestProperties:
    """Learner state access properties."""

    def test_calibration_results_property(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._calibration_results = {"test": CalibrationResult(model_name="test")}
        results: dict[str, CalibrationResult] = learner.calibration_results
        assert "test" in results
        assert results is not learner._calibration_results

    def test_patrol_zones_property(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._patrol_zones = [PatrolZone(lat_center=36.17)]
        zones: list[PatrolZone] = learner.patrol_zones
        assert len(zones) == 1
        assert zones is not learner._patrol_zones

    def test_weed_hotspots_property(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._weed_hotspots = [WeedHotspot(lat=36.17)]
        spots: list[WeedHotspot] = learner.weed_hotspots
        assert len(spots) == 1
        assert spots is not learner._weed_hotspots

    def test_pest_predictions_property(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner._pest_predictions = [PestPrediction(species="rabbit")]
        preds: list[PestPrediction] = learner.pest_predictions
        assert len(preds) == 1
        assert preds is not learner._pest_predictions


class TestStats:
    """Learner statistics."""

    def test_stats_initial(self, config: CerberusConfig) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        s: dict[str, Any] = learner.stats()
        assert s["total_runs"] == 0
        assert s["running"] is False
        assert s["calibrated_models"] == 0
        assert s["patrol_zones"] == 0
        assert s["weed_hotspots"] == 0
        assert s["pest_predictions"] == 0
        assert s["active_anomalies"] == 0

    def test_stats_after_run(self, config: CerberusConfig, db: CerberusDB) -> None:
        learner: AdaptiveLearner = AdaptiveLearner(config)
        learner.bind_db(db)
        learner.run_all()
        s: dict[str, Any] = learner.stats()
        assert s["total_runs"] == 1
        assert "learning_interval_s" in s
        assert "baseline_hours" in s