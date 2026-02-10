"""
Cerberus Obstacle Detection System
Three HC-SR04 ultrasonic sensors (front-center, left-front, right-front)
provide real-time distance measurements for obstacle avoidance. Background
polling thread maintains a live distance map. The drive controller checks
this before executing any forward movement.

Behavior Tiers:
    - Clear zone (>100cm): Full speed, normal operation
    - Warning zone (60-100cm): Reduce speed, increase scan rate
    - Caution zone (30-60cm): Stop forward motion, attempt lateral avoidance
    - Critical zone (<30cm): Emergency stop, alert, wait for clear path

Hardware: HC-SR04 ultrasonic sensors
    - Operating voltage: 5V (trigger/echo are 5V logic)
    - IMPORTANT: Echo pin returns 5V — use a voltage divider (1K + 2K)
      to bring it down to 3.3V safe for Pi 5 GPIO
    - Range: 2cm - 400cm
    - Effective angle: ~15 degrees
    - Trigger: 10us pulse on trigger pin
    - Echo: pulse width proportional to distance
"""

import time
import math
import logging
import threading
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)

SPEED_OF_SOUND_CM_PER_US: float = 0.0343
MAX_RANGE_CM: float = 400.0
MIN_RANGE_CM: float = 2.0
TIMEOUT_S: float = 0.03


class ObstacleZone(Enum):
    """Distance-based behavior zones."""
    CLEAR = "clear"
    WARNING = "warning"
    CAUTION = "caution"
    CRITICAL = "critical"


class AvoidanceDirection(Enum):
    """Suggested avoidance direction."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    REVERSE = "reverse"
    STOP = "stop"


@dataclass
class SensorReading:
    """A single ultrasonic sensor measurement."""
    name: str = ""
    distance_cm: float = MAX_RANGE_CM
    valid: bool = True
    timestamp: float = 0.0

    @property
    def zone(self) -> ObstacleZone:
        if self.distance_cm >= 100.0:
            return ObstacleZone.CLEAR
        elif self.distance_cm >= 60.0:
            return ObstacleZone.WARNING
        elif self.distance_cm >= 30.0:
            return ObstacleZone.CAUTION
        else:
            return ObstacleZone.CRITICAL


@dataclass
class ObstacleMap:
    """Real-time obstacle awareness from all sensors."""
    front: SensorReading = field(default_factory=lambda: SensorReading(name="front"))
    left_front: SensorReading = field(default_factory=lambda: SensorReading(name="left_front"))
    right_front: SensorReading = field(default_factory=lambda: SensorReading(name="right_front"))
    timestamp: float = 0.0

    @property
    def closest_distance_cm(self) -> float:
        distances: list[float] = [
            s.distance_cm for s in [self.front, self.left_front, self.right_front] if s.valid
        ]
        return min(distances) if distances else MAX_RANGE_CM

    @property
    def closest_sensor(self) -> SensorReading:
        sensors: list[SensorReading] = [self.front, self.left_front, self.right_front]
        valid_sensors: list[SensorReading] = [s for s in sensors if s.valid]
        if not valid_sensors:
            return self.front
        return min(valid_sensors, key=lambda s: s.distance_cm)

    @property
    def overall_zone(self) -> ObstacleZone:
        zones: list[ObstacleZone] = [
            s.zone for s in [self.front, self.left_front, self.right_front] if s.valid
        ]
        if not zones:
            return ObstacleZone.CLEAR

        priority: dict[ObstacleZone, int] = {
            ObstacleZone.CRITICAL: 3,
            ObstacleZone.CAUTION: 2,
            ObstacleZone.WARNING: 1,
            ObstacleZone.CLEAR: 0
        }
        return max(zones, key=lambda z: priority[z])

    @property
    def path_clear(self) -> bool:
        return self.overall_zone in (ObstacleZone.CLEAR, ObstacleZone.WARNING)

    @property
    def all_valid(self) -> bool:
        return self.front.valid and self.left_front.valid and self.right_front.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_cm": round(self.front.distance_cm, 1),
            "left_front_cm": round(self.left_front.distance_cm, 1),
            "right_front_cm": round(self.right_front.distance_cm, 1),
            "closest_cm": round(self.closest_distance_cm, 1),
            "overall_zone": self.overall_zone.value,
            "path_clear": self.path_clear,
            "all_valid": self.all_valid,
            "timestamp": self.timestamp
        }


@dataclass
class AvoidanceRecommendation:
    """Recommended action based on obstacle map analysis."""
    direction: AvoidanceDirection = AvoidanceDirection.NONE
    zone: ObstacleZone = ObstacleZone.CLEAR
    speed_limit: float = 1.0
    reason: str = ""
    obstacle_map: Optional[ObstacleMap] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "zone": self.zone.value,
            "speed_limit": round(self.speed_limit, 2),
            "reason": self.reason
        }


class ObstacleDetector:
    """
    Manages three HC-SR04 ultrasonic sensors for obstacle detection.
    Runs a background polling thread that continuously updates the
    obstacle map. The drive controller queries this before movement.

    Sensor Placement:
        - Front-center: straight ahead (0 degrees)
        - Left-front: 45 degrees left of center
        - Right-front: 45 degrees right of center

    Wiring per sensor (3.3V safe for Pi 5):
        - VCC -> 5V
        - GND -> GND
        - Trigger -> GPIO (direct, 3.3V is enough to trigger)
        - Echo -> Voltage divider (1K + 2K) -> GPIO
          Echo is 5V output! Without divider you WILL damage the Pi 5.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._enabled: bool = config.get("obstacle", "enabled", default=True)

        self._front_trigger: int = config.get("obstacle", "front", "trigger_pin", default=17)
        self._front_echo: int = config.get("obstacle", "front", "echo_pin", default=27)
        self._left_trigger: int = config.get("obstacle", "left_front", "trigger_pin", default=22)
        self._left_echo: int = config.get("obstacle", "left_front", "echo_pin", default=10)
        self._right_trigger: int = config.get("obstacle", "right_front", "trigger_pin", default=9)
        self._right_echo: int = config.get("obstacle", "right_front", "echo_pin", default=11)

        self._poll_interval: float = config.get("obstacle", "poll_interval", default=0.1)
        self._median_samples: int = config.get("obstacle", "median_samples", default=3)
        self._clear_threshold_cm: float = config.get("obstacle", "clear_threshold_cm", default=100.0)
        self._warning_threshold_cm: float = config.get("obstacle", "warning_threshold_cm", default=60.0)
        self._caution_threshold_cm: float = config.get("obstacle", "caution_threshold_cm", default=30.0)
        self._warning_speed_limit: float = config.get("obstacle", "warning_speed_limit", default=0.5)
        self._caution_speed_limit: float = config.get("obstacle", "caution_speed_limit", default=0.0)

        self._obstacle_map: ObstacleMap = ObstacleMap()
        self._lock: threading.Lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._hardware_available: bool = False
        self._sensors: dict[str, Any] = {}

        self._on_zone_change: Optional[Any] = None
        self._last_zone: ObstacleZone = ObstacleZone.CLEAR

        self._total_readings: int = 0
        self._failed_readings: int = 0

        if self._enabled:
            self._init_hardware()

    def _init_hardware(self) -> None:
        """Initialize HC-SR04 sensors via gpiozero DistanceSensor."""
        try:
            from gpiozero import DistanceSensor

            self._sensors["front"] = DistanceSensor(
                echo=self._front_echo,
                trigger=self._front_trigger,
                max_distance=MAX_RANGE_CM / 100.0,
                threshold_distance=self._caution_threshold_cm / 100.0
            )
            self._sensors["left_front"] = DistanceSensor(
                echo=self._left_echo,
                trigger=self._left_trigger,
                max_distance=MAX_RANGE_CM / 100.0,
                threshold_distance=self._caution_threshold_cm / 100.0
            )
            self._sensors["right_front"] = DistanceSensor(
                echo=self._right_echo,
                trigger=self._right_trigger,
                max_distance=MAX_RANGE_CM / 100.0,
                threshold_distance=self._caution_threshold_cm / 100.0
            )

            self._hardware_available = True
            logger.info(
                "Obstacle sensors initialized — front(%d/%d), left(%d/%d), right(%d/%d)",
                self._front_trigger, self._front_echo,
                self._left_trigger, self._left_echo,
                self._right_trigger, self._right_echo
            )

        except ImportError:
            self._hardware_available = False
            logger.warning("gpiozero not available — obstacle detection in simulation mode")
        except Exception as e:
            self._hardware_available = False
            logger.error("Failed to initialize obstacle sensors: %s", e)

    def start(self) -> bool:
        """Start background polling thread."""
        if not self._enabled:
            logger.info("Obstacle detection disabled in config")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="obstacle-detector",
            daemon=True
        )
        self._thread.start()
        logger.info("Obstacle detection started (interval=%.2fs, samples=%d)",
                     self._poll_interval, self._median_samples)
        return True

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        for name, sensor in self._sensors.items():
            try:
                sensor.close()
            except Exception:
                pass
        self._sensors.clear()

        logger.info(
            "Obstacle detection stopped — %d readings, %d failures (%.1f%% success)",
            self._total_readings,
            self._failed_readings,
            (1.0 - self._failed_readings / max(1, self._total_readings)) * 100
        )

    def _poll_loop(self) -> None:
        """Background thread that continuously polls sensors."""
        while self._running:
            try:
                self._update_readings()
            except Exception as e:
                logger.error("Obstacle poll error: %s", e)

            time.sleep(self._poll_interval)

    def _update_readings(self) -> None:
        """Read all sensors and update the obstacle map."""
        now: float = time.time()

        front: SensorReading = self._read_sensor("front", self._median_samples)
        left: SensorReading = self._read_sensor("left_front", self._median_samples)
        right: SensorReading = self._read_sensor("right_front", self._median_samples)

        with self._lock:
            self._obstacle_map = ObstacleMap(
                front=front,
                left_front=left,
                right_front=right,
                timestamp=now
            )

        new_zone: ObstacleZone = self._obstacle_map.overall_zone
        if new_zone != self._last_zone:
            self._last_zone = new_zone
            logger.info(
                "Obstacle zone change: %s (front=%.0fcm, left=%.0fcm, right=%.0fcm)",
                new_zone.value, front.distance_cm, left.distance_cm, right.distance_cm
            )
            if self._on_zone_change is not None:
                try:
                    self._on_zone_change(new_zone, self._obstacle_map)
                except Exception as e:
                    logger.error("Zone change callback error: %s", e)

    def _read_sensor(self, name: str, samples: int = 1) -> SensorReading:
        """Read a single sensor with optional median filtering."""
        self._total_readings += 1

        if not self._hardware_available or name not in self._sensors:
            return self._simulated_reading(name)

        readings: list[float] = []
        for _ in range(samples):
            try:
                distance_m: float = self._sensors[name].distance
                distance_cm: float = distance_m * 100.0

                if MIN_RANGE_CM <= distance_cm <= MAX_RANGE_CM:
                    readings.append(distance_cm)
            except Exception:
                pass

            if samples > 1:
                time.sleep(0.01)

        if not readings:
            self._failed_readings += 1
            return SensorReading(
                name=name,
                distance_cm=MAX_RANGE_CM,
                valid=False,
                timestamp=time.time()
            )

        readings.sort()
        median_idx: int = len(readings) // 2
        distance: float = readings[median_idx]

        return SensorReading(
            name=name,
            distance_cm=distance,
            valid=True,
            timestamp=time.time()
        )

    def _simulated_reading(self, name: str) -> SensorReading:
        """Simulated reading for dev environment."""
        import random
        base: float = 150.0
        noise: float = random.uniform(-20.0, 20.0)
        return SensorReading(
            name=name,
            distance_cm=max(MIN_RANGE_CM, base + noise),
            valid=True,
            timestamp=time.time()
        )

    def get_obstacle_map(self) -> ObstacleMap:
        """Get the current obstacle map. Thread-safe."""
        with self._lock:
            return self._obstacle_map

    def is_path_clear(self) -> bool:
        """Check if the forward path is clear for driving."""
        with self._lock:
            return self._obstacle_map.path_clear

    def closest_obstacle(self) -> float:
        """Get distance to the closest obstacle in cm."""
        with self._lock:
            return self._obstacle_map.closest_distance_cm

    def current_zone(self) -> ObstacleZone:
        """Get the current obstacle zone."""
        with self._lock:
            return self._obstacle_map.overall_zone

    def get_avoidance_recommendation(self) -> AvoidanceRecommendation:
        """
        Analyze the obstacle map and recommend an avoidance action.
        The drive controller calls this before executing movement.
        """
        with self._lock:
            obs: ObstacleMap = self._obstacle_map

        zone: ObstacleZone = obs.overall_zone

        if zone == ObstacleZone.CLEAR:
            return AvoidanceRecommendation(
                direction=AvoidanceDirection.NONE,
                zone=zone,
                speed_limit=1.0,
                reason="Path clear",
                obstacle_map=obs
            )

        if zone == ObstacleZone.WARNING:
            return AvoidanceRecommendation(
                direction=AvoidanceDirection.NONE,
                zone=zone,
                speed_limit=self._warning_speed_limit,
                reason=f"Obstacle warning — closest {obs.closest_distance_cm:.0f}cm",
                obstacle_map=obs
            )

        front_clear: bool = obs.front.zone in (ObstacleZone.CLEAR, ObstacleZone.WARNING)
        left_clear: bool = obs.left_front.zone in (ObstacleZone.CLEAR, ObstacleZone.WARNING)
        right_clear: bool = obs.right_front.zone in (ObstacleZone.CLEAR, ObstacleZone.WARNING)

        if zone == ObstacleZone.CAUTION:
            if front_clear:
                return AvoidanceRecommendation(
                    direction=AvoidanceDirection.NONE,
                    zone=zone,
                    speed_limit=self._caution_speed_limit,
                    reason="Front clear but side obstacle — proceed with caution",
                    obstacle_map=obs
                )

            if left_clear and not right_clear:
                return AvoidanceRecommendation(
                    direction=AvoidanceDirection.LEFT,
                    zone=zone,
                    speed_limit=self._caution_speed_limit,
                    reason=f"Front blocked ({obs.front.distance_cm:.0f}cm) — left clear ({obs.left_front.distance_cm:.0f}cm)",
                    obstacle_map=obs
                )

            if right_clear and not left_clear:
                return AvoidanceRecommendation(
                    direction=AvoidanceDirection.RIGHT,
                    zone=zone,
                    speed_limit=self._caution_speed_limit,
                    reason=f"Front blocked ({obs.front.distance_cm:.0f}cm) — right clear ({obs.right_front.distance_cm:.0f}cm)",
                    obstacle_map=obs
                )

            if left_clear and right_clear:
                if obs.left_front.distance_cm >= obs.right_front.distance_cm:
                    direction: AvoidanceDirection = AvoidanceDirection.LEFT
                else:
                    direction = AvoidanceDirection.RIGHT
                return AvoidanceRecommendation(
                    direction=direction,
                    zone=zone,
                    speed_limit=self._caution_speed_limit,
                    reason=f"Front blocked — choosing {direction.value} (more clearance)",
                    obstacle_map=obs
                )

            return AvoidanceRecommendation(
                direction=AvoidanceDirection.REVERSE,
                zone=zone,
                speed_limit=self._caution_speed_limit,
                reason="All forward paths blocked — reverse recommended",
                obstacle_map=obs
            )

        return AvoidanceRecommendation(
            direction=AvoidanceDirection.STOP,
            zone=zone,
            speed_limit=0.0,
            reason=f"CRITICAL — obstacle at {obs.closest_distance_cm:.0f}cm — emergency stop",
            obstacle_map=obs
        )

    def set_zone_change_callback(self, callback: Any) -> None:
        """Register a callback for obstacle zone changes. Signature: callback(zone, obstacle_map)."""
        self._on_zone_change = callback

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def hardware_available(self) -> bool:
        return self._hardware_available

    @property
    def total_readings(self) -> int:
        return self._total_readings

    @property
    def failure_rate(self) -> float:
        if self._total_readings == 0:
            return 0.0
        return self._failed_readings / self._total_readings