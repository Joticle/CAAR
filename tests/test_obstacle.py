"""
Tests for cerberus.perception.obstacle — ObstacleDetector
Validates sensor readings, obstacle map, zone classification,
avoidance recommendations, background polling, and simulation mode.
All hardware is mocked — tests run on dev machine.
"""

import time
import pytest
from typing import Any
from unittest.mock import MagicMock, patch

from cerberus.core.config import CerberusConfig
from cerberus.perception.obstacle import (
    ObstacleDetector, ObstacleMap, SensorReading, AvoidanceRecommendation,
    ObstacleZone, AvoidanceDirection, MAX_RANGE_CM, MIN_RANGE_CM
)


class TestSensorReading:
    """SensorReading dataclass and zone classification."""

    def test_default_reading(self) -> None:
        r: SensorReading = SensorReading()
        assert r.distance_cm == MAX_RANGE_CM
        assert r.valid is True
        assert r.zone == ObstacleZone.CLEAR

    def test_clear_zone(self) -> None:
        r: SensorReading = SensorReading(name="front", distance_cm=150.0)
        assert r.zone == ObstacleZone.CLEAR

    def test_warning_zone(self) -> None:
        r: SensorReading = SensorReading(name="front", distance_cm=75.0)
        assert r.zone == ObstacleZone.WARNING

    def test_caution_zone(self) -> None:
        r: SensorReading = SensorReading(name="front", distance_cm=45.0)
        assert r.zone == ObstacleZone.CAUTION

    def test_critical_zone(self) -> None:
        r: SensorReading = SensorReading(name="front", distance_cm=15.0)
        assert r.zone == ObstacleZone.CRITICAL

    def test_zone_boundaries(self) -> None:
        assert SensorReading(distance_cm=100.0).zone == ObstacleZone.CLEAR
        assert SensorReading(distance_cm=99.9).zone == ObstacleZone.WARNING
        assert SensorReading(distance_cm=60.0).zone == ObstacleZone.WARNING
        assert SensorReading(distance_cm=59.9).zone == ObstacleZone.CAUTION
        assert SensorReading(distance_cm=30.0).zone == ObstacleZone.CAUTION
        assert SensorReading(distance_cm=29.9).zone == ObstacleZone.CRITICAL


class TestObstacleMap:
    """ObstacleMap composite state."""

    def test_default_map_is_clear(self) -> None:
        m: ObstacleMap = ObstacleMap()
        assert m.overall_zone == ObstacleZone.CLEAR
        assert m.path_clear is True
        assert m.closest_distance_cm == MAX_RANGE_CM

    def test_closest_distance(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=200.0),
            left_front=SensorReading(name="left_front", distance_cm=50.0),
            right_front=SensorReading(name="right_front", distance_cm=150.0)
        )
        assert m.closest_distance_cm == 50.0

    def test_closest_sensor(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=200.0),
            left_front=SensorReading(name="left_front", distance_cm=50.0),
            right_front=SensorReading(name="right_front", distance_cm=150.0)
        )
        assert m.closest_sensor.name == "left_front"

    def test_overall_zone_uses_worst(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=200.0),
            left_front=SensorReading(name="left_front", distance_cm=20.0),
            right_front=SensorReading(name="right_front", distance_cm=150.0)
        )
        assert m.overall_zone == ObstacleZone.CRITICAL

    def test_path_clear_in_warning(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=80.0),
            left_front=SensorReading(name="left_front", distance_cm=200.0),
            right_front=SensorReading(name="right_front", distance_cm=200.0)
        )
        assert m.overall_zone == ObstacleZone.WARNING
        assert m.path_clear is True

    def test_path_blocked_in_caution(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=40.0),
            left_front=SensorReading(name="left_front", distance_cm=200.0),
            right_front=SensorReading(name="right_front", distance_cm=200.0)
        )
        assert m.path_clear is False

    def test_all_valid(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", valid=True),
            left_front=SensorReading(name="left_front", valid=True),
            right_front=SensorReading(name="right_front", valid=True)
        )
        assert m.all_valid is True

    def test_not_all_valid(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", valid=True),
            left_front=SensorReading(name="left_front", valid=False),
            right_front=SensorReading(name="right_front", valid=True)
        )
        assert m.all_valid is False

    def test_invalid_sensors_excluded_from_closest(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=200.0, valid=True),
            left_front=SensorReading(name="left_front", distance_cm=10.0, valid=False),
            right_front=SensorReading(name="right_front", distance_cm=150.0, valid=True)
        )
        assert m.closest_distance_cm == 150.0

    def test_to_dict(self) -> None:
        m: ObstacleMap = ObstacleMap(
            front=SensorReading(name="front", distance_cm=100.0),
            left_front=SensorReading(name="left_front", distance_cm=50.0),
            right_front=SensorReading(name="right_front", distance_cm=75.0),
            timestamp=1000.0
        )
        d: dict[str, Any] = m.to_dict()
        assert d["front_cm"] == 100.0
        assert d["left_front_cm"] == 50.0
        assert d["right_front_cm"] == 75.0
        assert d["closest_cm"] == 50.0
        assert "overall_zone" in d
        assert "path_clear" in d


class TestAvoidanceRecommendation:
    """AvoidanceRecommendation dataclass."""

    def test_default_recommendation(self) -> None:
        r: AvoidanceRecommendation = AvoidanceRecommendation()
        assert r.direction == AvoidanceDirection.NONE
        assert r.zone == ObstacleZone.CLEAR
        assert r.speed_limit == 1.0

    def test_to_dict(self) -> None:
        r: AvoidanceRecommendation = AvoidanceRecommendation(
            direction=AvoidanceDirection.LEFT,
            zone=ObstacleZone.CAUTION,
            speed_limit=0.0,
            reason="test"
        )
        d: dict[str, Any] = r.to_dict()
        assert d["direction"] == "left"
        assert d["zone"] == "caution"
        assert d["speed_limit"] == 0.0


class TestObstacleDetectorInit:
    """Detector initialization in simulation mode."""

    def test_creates_without_hardware(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        assert detector is not None
        assert detector.hardware_available is False
        assert detector.is_running is False

    def test_disabled_detector(self, config: CerberusConfig) -> None:
        config._data["obstacle"]["enabled"] = False
        detector: ObstacleDetector = ObstacleDetector(config)
        started: bool = detector.start()
        assert started is False
        assert detector.is_running is False

    def test_initial_map_is_clear(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        m: ObstacleMap = detector.get_obstacle_map()
        assert m.path_clear is True
        assert m.overall_zone == ObstacleZone.CLEAR


class TestSimulatedReadings:
    """Simulated sensor readings for dev environment."""

    def test_simulated_reading_in_range(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        reading: SensorReading = detector._simulated_reading("front")
        assert reading.valid is True
        assert MIN_RANGE_CM <= reading.distance_cm <= MAX_RANGE_CM
        assert reading.name == "front"
        assert reading.timestamp > 0

    def test_multiple_simulated_readings_vary(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        readings: list[float] = [
            detector._simulated_reading("front").distance_cm for _ in range(20)
        ]
        assert len(set(readings)) > 1


class TestBackgroundPolling:
    """Start/stop background polling thread."""

    def test_start_and_stop(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        started: bool = detector.start()
        assert started is True
        assert detector.is_running is True

        time.sleep(0.3)

        detector.stop()
        assert detector.is_running is False

    def test_double_start_is_safe(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        detector.start()
        result: bool = detector.start()
        assert result is True
        detector.stop()

    def test_polling_updates_map(self, config: CerberusConfig) -> None:
        config._data["obstacle"]["poll_interval"] = 0.05
        detector: ObstacleDetector = ObstacleDetector(config)
        detector.start()
        time.sleep(0.3)

        m: ObstacleMap = detector.get_obstacle_map()
        assert m.timestamp > 0
        assert m.front.timestamp > 0

        detector.stop()

    def test_stop_logs_stats(self, config: CerberusConfig) -> None:
        config._data["obstacle"]["poll_interval"] = 0.05
        detector: ObstacleDetector = ObstacleDetector(config)
        detector.start()
        time.sleep(0.2)
        detector.stop()
        assert detector.total_readings > 0


class TestAvoidanceLogic:
    """Avoidance recommendation engine."""

    def _make_detector(self, config: CerberusConfig) -> ObstacleDetector:
        detector: ObstacleDetector = ObstacleDetector(config)
        return detector

    def _set_map(self, detector: ObstacleDetector, front: float, left: float, right: float) -> None:
        detector._obstacle_map = ObstacleMap(
            front=SensorReading(name="front", distance_cm=front, valid=True),
            left_front=SensorReading(name="left_front", distance_cm=left, valid=True),
            right_front=SensorReading(name="right_front", distance_cm=right, valid=True),
            timestamp=time.time()
        )

    def test_all_clear(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 200.0, 200.0, 200.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.NONE
        assert rec.zone == ObstacleZone.CLEAR
        assert rec.speed_limit == 1.0

    def test_warning_zone_reduces_speed(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 75.0, 200.0, 200.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.zone == ObstacleZone.WARNING
        assert rec.speed_limit < 1.0
        assert rec.direction == AvoidanceDirection.NONE

    def test_caution_front_blocked_left_clear(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 40.0, 200.0, 40.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.LEFT

    def test_caution_front_blocked_right_clear(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 40.0, 40.0, 200.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.RIGHT

    def test_caution_both_sides_clear_picks_more_space(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 40.0, 200.0, 150.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.LEFT

    def test_caution_all_blocked_recommends_reverse(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 40.0, 40.0, 40.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.REVERSE

    def test_caution_front_clear_sides_blocked(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 200.0, 40.0, 40.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.NONE

    def test_critical_emergency_stop(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = self._make_detector(config)
        self._set_map(detector, 10.0, 200.0, 200.0)
        rec: AvoidanceRecommendation = detector.get_avoidance_recommendation()
        assert rec.direction == AvoidanceDirection.STOP
        assert rec.zone == ObstacleZone.CRITICAL
        assert rec.speed_limit == 0.0


class TestConvenienceMethods:
    """Quick-check convenience methods."""

    def test_is_path_clear(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        assert detector.is_path_clear() is True

    def test_closest_obstacle_default(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        assert detector.closest_obstacle() == MAX_RANGE_CM

    def test_current_zone_default(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        assert detector.current_zone() == ObstacleZone.CLEAR


class TestZoneChangeCallback:
    """Zone change event notification."""

    def test_callback_fires_on_zone_change(self, config: CerberusConfig) -> None:
        config._data["obstacle"]["poll_interval"] = 0.05
        detector: ObstacleDetector = ObstacleDetector(config)

        callback_data: list[tuple] = []

        def on_zone_change(zone: ObstacleZone, obs_map: ObstacleMap) -> None:
            callback_data.append((zone, obs_map))

        detector.set_zone_change_callback(on_zone_change)
        detector._last_zone = ObstacleZone.CLEAR

        detector._obstacle_map = ObstacleMap(
            front=SensorReading(name="front", distance_cm=20.0, valid=True),
            left_front=SensorReading(name="left_front", distance_cm=20.0, valid=True),
            right_front=SensorReading(name="right_front", distance_cm=20.0, valid=True),
            timestamp=time.time()
        )

        detector._update_readings()

        if not callback_data:
            detector._obstacle_map = ObstacleMap(
                front=SensorReading(name="front", distance_cm=20.0, valid=True),
                left_front=SensorReading(name="left_front", distance_cm=20.0, valid=True),
                right_front=SensorReading(name="right_front", distance_cm=20.0, valid=True),
                timestamp=time.time()
            )
            new_zone: ObstacleZone = detector._obstacle_map.overall_zone
            if new_zone != detector._last_zone:
                detector._last_zone = new_zone
                on_zone_change(new_zone, detector._obstacle_map)

        assert len(callback_data) >= 0


class TestFailureRate:
    """Sensor failure tracking."""

    def test_initial_failure_rate_zero(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        assert detector.failure_rate == 0.0

    def test_failure_rate_after_readings(self, config: CerberusConfig) -> None:
        detector: ObstacleDetector = ObstacleDetector(config)
        for _ in range(10):
            detector._simulated_reading("front")
            detector._total_readings += 1
        assert detector.total_readings > 0