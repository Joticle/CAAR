"""
Cerberus GPS Waypoint Navigator
Autonomous point-to-point navigation using GPS coordinates.
Calculates bearing, distance, and heading corrections to drive
Cerberus between waypoints without human input.
"""

import math
import time
import logging
import threading
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig
from cerberus.mobility.drive_controller import DriveController


logger: logging.Logger = logging.getLogger(__name__)

EARTH_RADIUS_M: float = 6371000.0


class NavState(Enum):
    """Navigation state machine."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    TURNING = "turning"
    ARRIVING = "arriving"
    ARRIVED = "arrived"
    PAUSED = "paused"
    BLOCKED = "blocked"
    NO_FIX = "no_fix"
    ERROR = "error"


@dataclass
class Waypoint:
    """A GPS coordinate target."""
    lat: float
    lon: float
    name: str = ""
    radius_m: float = 2.0
    speed: float = 0.5
    action: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "name": self.name,
            "radius_m": self.radius_m,
            "speed": self.speed,
            "action": self.action,
            "metadata": self.metadata
        }


@dataclass
class NavStatus:
    """Current navigation status snapshot."""
    state: NavState = NavState.IDLE
    current_lat: float = 0.0
    current_lon: float = 0.0
    target_lat: float = 0.0
    target_lon: float = 0.0
    distance_m: float = 0.0
    bearing_deg: float = 0.0
    heading_deg: float = 0.0
    heading_error_deg: float = 0.0
    waypoint_index: int = 0
    waypoint_count: int = 0
    speed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "current_lat": round(self.current_lat, 7),
            "current_lon": round(self.current_lon, 7),
            "target_lat": round(self.target_lat, 7),
            "target_lon": round(self.target_lon, 7),
            "distance_m": round(self.distance_m, 2),
            "bearing_deg": round(self.bearing_deg, 1),
            "heading_deg": round(self.heading_deg, 1),
            "heading_error_deg": round(self.heading_error_deg, 1),
            "waypoint_index": self.waypoint_index,
            "waypoint_count": self.waypoint_count,
            "speed": round(self.speed, 2)
        }


class Navigator:
    """
    GPS waypoint navigator for Cerberus.
    Accepts a list of waypoints and autonomously drives between them.
    Uses bearing calculation and heading correction via skid-steer turning.

    GPS position comes from an external GPS provider (perception/gps.py).
    Heading is derived from consecutive GPS positions (no magnetometer).
    """

    def __init__(
        self,
        drive: DriveController,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._drive: DriveController = drive
        self._config: CerberusConfig = config

        self._waypoint_radius: float = config.get("navigation", "waypoint_radius_m", default=2.0)
        self._heading_tolerance: float = config.get("navigation", "heading_tolerance_deg", default=15)
        self._max_speed: float = config.get("navigation", "max_speed", default=0.7)
        self._turn_speed: float = config.get("navigation", "turn_speed", default=0.4)
        self._gps_timeout: int = config.get("navigation", "gps_timeout_seconds", default=10)

        self._waypoints: list[Waypoint] = []
        self._current_index: int = 0
        self._state: NavState = NavState.IDLE

        self._current_lat: float = 0.0
        self._current_lon: float = 0.0
        self._current_heading: float = 0.0
        self._has_fix: bool = False
        self._last_gps_time: float = 0.0

        self._prev_lat: float = 0.0
        self._prev_lon: float = 0.0

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()

        self._nav_interval: float = 0.2
        self._min_heading_distance: float = 0.5

        self._on_waypoint_reached: Optional[callable] = None
        self._on_route_complete: Optional[callable] = None
        self._on_nav_error: Optional[callable] = None

        logger.info(
            "Navigator created — radius=%.1fm, heading_tolerance=%.1f°, max_speed=%.1f",
            self._waypoint_radius, self._heading_tolerance, self._max_speed
        )

    def update_position(self, lat: float, lon: float) -> None:
        """
        Called by GPS provider with updated position.
        Also derives heading from consecutive positions.
        """
        with self._lock:
            if self._has_fix and self._current_lat != 0.0:
                distance: float = self._haversine(
                    self._current_lat, self._current_lon, lat, lon
                )
                if distance >= self._min_heading_distance:
                    self._current_heading = self._bearing(
                        self._current_lat, self._current_lon, lat, lon
                    )
                    self._prev_lat = self._current_lat
                    self._prev_lon = self._current_lon

            self._current_lat = lat
            self._current_lon = lon
            self._has_fix = True
            self._last_gps_time = time.time()

    def set_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Load a route of waypoints to navigate."""
        with self._lock:
            self._waypoints = list(waypoints)
            self._current_index = 0
            self._state = NavState.IDLE

        logger.info("Route loaded: %d waypoints", len(waypoints))
        for i, wp in enumerate(waypoints):
            logger.debug("  WP %d: %s (%.7f, %.7f) radius=%.1fm",
                         i, wp.name or f"wp_{i}", wp.lat, wp.lon, wp.radius_m)

    def add_waypoint(self, waypoint: Waypoint) -> None:
        """Add a single waypoint to the end of the route."""
        with self._lock:
            self._waypoints.append(waypoint)
        logger.info("Waypoint added: %s (%.7f, %.7f)", waypoint.name, waypoint.lat, waypoint.lon)

    def clear_waypoints(self) -> None:
        """Clear all waypoints and stop navigation."""
        self.stop_navigation()
        with self._lock:
            self._waypoints.clear()
            self._current_index = 0
        logger.info("Waypoints cleared")

    def start_navigation(self) -> bool:
        """Begin navigating the loaded waypoint route."""
        with self._lock:
            if not self._waypoints:
                logger.warning("Cannot start navigation — no waypoints loaded")
                return False
            if self._running:
                logger.warning("Navigation already running")
                return False
            if not self._has_fix:
                logger.warning("Cannot start navigation — no GPS fix")
                self._state = NavState.NO_FIX
                return False

        self._running = True
        self._state = NavState.NAVIGATING
        self._thread = threading.Thread(
            target=self._nav_loop,
            name="navigator",
            daemon=True
        )
        self._thread.start()

        logger.info("Navigation started — %d waypoints", len(self._waypoints))
        return True

    def stop_navigation(self) -> None:
        """Stop navigation and halt the rover."""
        if not self._running:
            return

        self._running = False
        self._pause_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Navigator thread did not stop cleanly")
            self._thread = None

        self._drive.stop()

        with self._lock:
            self._state = NavState.IDLE

        logger.info("Navigation stopped")

    def pause(self) -> None:
        """Pause navigation — rover stops but route is preserved."""
        self._pause_event.clear()
        self._drive.stop()
        with self._lock:
            self._state = NavState.PAUSED
        logger.info("Navigation paused at waypoint %d", self._current_index)

    def resume(self) -> None:
        """Resume navigation from where it was paused."""
        with self._lock:
            self._state = NavState.NAVIGATING
        self._pause_event.set()
        logger.info("Navigation resumed")

    def skip_waypoint(self) -> None:
        """Skip the current waypoint and move to the next."""
        with self._lock:
            if self._current_index < len(self._waypoints) - 1:
                self._current_index += 1
                logger.info("Skipped to waypoint %d", self._current_index)
            else:
                logger.warning("Cannot skip — already on last waypoint")

    def _nav_loop(self) -> None:
        """Main navigation loop — runs in background thread."""
        logger.info("Navigator loop started")

        while self._running:
            self._pause_event.wait()

            if not self._running:
                break

            try:
                with self._lock:
                    if self._current_index >= len(self._waypoints):
                        self._state = NavState.ARRIVED
                        self._running = False
                        self._drive.stop()
                        logger.info("Route complete — all waypoints reached")
                        if self._on_route_complete:
                            try:
                                self._on_route_complete()
                            except Exception as e:
                                logger.error("Route complete callback error: %s", e)
                        break

                if not self._check_gps():
                    time.sleep(self._nav_interval)
                    continue

                self._navigate_to_current_waypoint()

            except Exception as e:
                logger.error("Navigation error: %s", e)
                with self._lock:
                    self._state = NavState.ERROR
                self._drive.stop()
                if self._on_nav_error:
                    try:
                        self._on_nav_error(str(e))
                    except Exception:
                        pass
                time.sleep(1.0)

            time.sleep(self._nav_interval)

        logger.info("Navigator loop exited")

    def _check_gps(self) -> bool:
        """Verify GPS fix is current."""
        with self._lock:
            if not self._has_fix:
                if self._state != NavState.NO_FIX:
                    self._state = NavState.NO_FIX
                    logger.warning("GPS fix lost — pausing navigation")
                self._drive.stop()
                return False

            elapsed: float = time.time() - self._last_gps_time
            if elapsed > self._gps_timeout:
                if self._state != NavState.NO_FIX:
                    self._state = NavState.NO_FIX
                    logger.warning("GPS data stale (%.1fs) — pausing navigation", elapsed)
                self._drive.stop()
                return False

        return True

    def _navigate_to_current_waypoint(self) -> None:
        """Execute one navigation cycle toward the current waypoint."""
        with self._lock:
            if self._current_index >= len(self._waypoints):
                return
            wp: Waypoint = self._waypoints[self._current_index]
            lat: float = self._current_lat
            lon: float = self._current_lon
            heading: float = self._current_heading

        distance: float = self._haversine(lat, lon, wp.lat, wp.lon)
        bearing: float = self._bearing(lat, lon, wp.lat, wp.lon)
        heading_error: float = self._normalize_angle(bearing - heading)

        arrival_radius: float = wp.radius_m if wp.radius_m > 0 else self._waypoint_radius

        if distance <= arrival_radius:
            self._on_waypoint_arrival(wp, distance)
            return

        speed: float = min(wp.speed, self._max_speed)

        if distance < arrival_radius * 3:
            speed *= 0.5

        if abs(heading_error) > self._heading_tolerance:
            with self._lock:
                self._state = NavState.TURNING

            if abs(heading_error) > 90:
                turn_value: float = 1.0 if heading_error > 0 else -1.0
                self._drive.drive(0.0, turn_value * self._turn_speed)
            else:
                turn_value = heading_error / 90.0
                turn_value = max(-1.0, min(1.0, turn_value))
                self._drive.drive(speed * 0.5, turn_value)
        else:
            with self._lock:
                self._state = NavState.NAVIGATING
            turn_correction: float = (heading_error / self._heading_tolerance) * 0.3
            turn_correction = max(-0.5, min(0.5, turn_correction))
            self._drive.drive(speed, turn_correction)

        logger.debug(
            "Nav: WP%d dist=%.1fm bearing=%.1f° heading=%.1f° error=%.1f° speed=%.2f",
            self._current_index, distance, bearing, heading, heading_error, speed
        )

    def _on_waypoint_arrival(self, wp: Waypoint, distance: float) -> None:
        """Handle arrival at a waypoint."""
        self._drive.stop()

        logger.info(
            "Waypoint %d reached: %s (%.7f, %.7f) — distance: %.2fm",
            self._current_index, wp.name or f"wp_{self._current_index}",
            wp.lat, wp.lon, distance
        )

        if self._on_waypoint_reached:
            try:
                self._on_waypoint_reached(self._current_index, wp)
            except Exception as e:
                logger.error("Waypoint reached callback error: %s", e)

        with self._lock:
            self._state = NavState.ARRIVING
            self._current_index += 1

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two GPS coordinates."""
        lat1_r: float = math.radians(lat1)
        lat2_r: float = math.radians(lat2)
        dlat: float = math.radians(lat2 - lat1)
        dlon: float = math.radians(lon2 - lon1)

        a: float = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_RADIUS_M * c

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing in degrees from point 1 to point 2."""
        lat1_r: float = math.radians(lat1)
        lat2_r: float = math.radians(lat2)
        dlon: float = math.radians(lon2 - lon1)

        x: float = math.sin(dlon) * math.cos(lat2_r)
        y: float = (
            math.cos(lat1_r) * math.sin(lat2_r) -
            math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        )

        bearing: float = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize an angle to -180 to +180 range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def register_waypoint_callback(self, callback: callable) -> None:
        """Register callback for waypoint arrival: callback(index, waypoint)."""
        self._on_waypoint_reached = callback

    def register_route_complete_callback(self, callback: callable) -> None:
        """Register callback for route completion: callback()."""
        self._on_route_complete = callback

    def register_error_callback(self, callback: callable) -> None:
        """Register callback for navigation errors: callback(error_message)."""
        self._on_nav_error = callback

    @property
    def status(self) -> NavStatus:
        """Get current navigation status."""
        with self._lock:
            target_lat: float = 0.0
            target_lon: float = 0.0
            distance: float = 0.0
            bearing: float = 0.0
            heading_error: float = 0.0

            if self._current_index < len(self._waypoints):
                wp: Waypoint = self._waypoints[self._current_index]
                target_lat = wp.lat
                target_lon = wp.lon
                if self._has_fix:
                    distance = self._haversine(
                        self._current_lat, self._current_lon, wp.lat, wp.lon
                    )
                    bearing = self._bearing(
                        self._current_lat, self._current_lon, wp.lat, wp.lon
                    )
                    heading_error = self._normalize_angle(bearing - self._current_heading)

            return NavStatus(
                state=self._state,
                current_lat=self._current_lat,
                current_lon=self._current_lon,
                target_lat=target_lat,
                target_lon=target_lon,
                distance_m=distance,
                bearing_deg=bearing,
                heading_deg=self._current_heading,
                heading_error_deg=heading_error,
                waypoint_index=self._current_index,
                waypoint_count=len(self._waypoints),
                speed=self._drive.state.speed
            )

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get the current target waypoint."""
        with self._lock:
            if self._current_index < len(self._waypoints):
                return self._waypoints[self._current_index]
            return None

    @property
    def is_navigating(self) -> bool:
        return self._running

    @property
    def nav_state(self) -> NavState:
        with self._lock:
            return self._state

    @property
    def waypoint_count(self) -> int:
        with self._lock:
            return len(self._waypoints)

    @property
    def current_position(self) -> tuple[float, float]:
        with self._lock:
            return (self._current_lat, self._current_lon)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"Navigator(state={self._state.value}, "
                f"wp={self._current_index}/{len(self._waypoints)}, "
                f"fix={'yes' if self._has_fix else 'no'})"
            )