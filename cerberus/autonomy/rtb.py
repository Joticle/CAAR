"""
Cerberus Return To Base (RTB)
Autonomously navigates Cerberus back to its home position.
Triggered by low battery, safety violations, mission completion,
or manual command. RTB overrides all other navigation — when Cerberus
needs to come home, it comes home.
"""

import time
import logging
import threading
from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

from cerberus.core.config import CerberusConfig
from cerberus.mobility.navigator import Navigator, Waypoint, NavState
from cerberus.mobility.drive_controller import DriveController


logger: logging.Logger = logging.getLogger(__name__)


class RTBState(Enum):
    """RTB execution states."""
    IDLE = "idle"
    INITIATED = "initiated"
    NAVIGATING = "navigating"
    APPROACHING = "approaching"
    ARRIVED = "arrived"
    DOCKING = "docking"
    SAFE_STOP = "safe_stop"
    FAILED = "failed"


class RTBReason(Enum):
    """Why RTB was triggered."""
    MANUAL = "manual"
    LOW_BATTERY = "low_battery"
    CRITICAL_BATTERY = "critical_battery"
    THERMAL = "thermal"
    SAFETY_VIOLATION = "safety_violation"
    MISSION_COMPLETE = "mission_complete"
    LOST_GPS = "lost_gps"
    REMOTE_COMMAND = "remote_command"
    SYSTEM_ERROR = "system_error"


@dataclass
class RTBStatus:
    """Current RTB status."""
    state: RTBState = RTBState.IDLE
    reason: RTBReason = RTBReason.MANUAL
    home_lat: float = 0.0
    home_lon: float = 0.0
    distance_to_home_m: float = 0.0
    elapsed_seconds: float = 0.0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason.value,
            "home_lat": round(self.home_lat, 7),
            "home_lon": round(self.home_lon, 7),
            "distance_to_home_m": round(self.distance_to_home_m, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "attempts": self.attempts
        }


class ReturnToBase:
    """
    Autonomous Return To Base for Cerberus.
    When triggered, aborts any active navigation or mission and
    drives directly to the home coordinates. If navigation fails,
    retries with increasing caution. If all retries fail, performs
    a safe stop in place and alerts.

    RTB has absolute authority over the drive system when active.
    Only the safety watchdog's emergency stop outranks RTB.
    """

    def __init__(
        self,
        navigator: Navigator,
        drive: DriveController,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._navigator: Navigator = navigator
        self._drive: DriveController = drive
        self._config: CerberusConfig = config

        self._home_lat: float = config.get("navigation", "home_lat", default=36.1699)
        self._home_lon: float = config.get("navigation", "home_lon", default=-115.1398)
        self._home_radius: float = config.get("navigation", "home_radius_m", default=1.5)
        self._approach_speed: float = config.get("navigation", "rtb_approach_speed", default=0.3)
        self._cruise_speed: float = config.get("navigation", "rtb_cruise_speed", default=0.5)
        self._max_retries: int = config.get("navigation", "rtb_max_retries", default=3)
        self._retry_delay: float = config.get("navigation", "rtb_retry_delay_seconds", default=10.0)
        self._gps_wait_timeout: float = config.get("navigation", "rtb_gps_wait_timeout", default=60.0)

        self._state: RTBState = RTBState.IDLE
        self._reason: RTBReason = RTBReason.MANUAL
        self._attempts: int = 0
        self._start_time: float = 0.0
        self._active: bool = False

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._waypoint_arrived: threading.Event = threading.Event()

        self._on_rtb_started: Optional[Callable[[RTBReason], None]] = None
        self._on_rtb_arrived: Optional[Callable[[], None]] = None
        self._on_rtb_failed: Optional[Callable[[str], None]] = None

        self._abort_mission_callback: Optional[Callable[[], None]] = None
        self._abort_patrol_callback: Optional[Callable[[], None]] = None
        self._abort_grid_callback: Optional[Callable[[], None]] = None

        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None

        logger.info(
            "RTB created — home=(%.7f, %.7f), radius=%.1fm, max_retries=%d",
            self._home_lat, self._home_lon, self._home_radius, self._max_retries
        )

    def bind_db(self, db: Any) -> None:
        """Bind database for RTB logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for RTB alerts."""
        self._mqtt = mqtt_client

    def register_arrived_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for successful RTB: callback()."""
        self._on_rtb_arrived = callback

    def register_failed_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for RTB failure: callback(reason)."""
        self._on_rtb_failed = callback

    def register_started_callback(self, callback: Callable[[RTBReason], None]) -> None:
        """Register callback for RTB start: callback(reason)."""
        self._on_rtb_started = callback

    def register_abort_mission(self, callback: Callable[[], None]) -> None:
        """Register callback to abort active mission."""
        self._abort_mission_callback = callback

    def register_abort_patrol(self, callback: Callable[[], None]) -> None:
        """Register callback to abort active patrol."""
        self._abort_patrol_callback = callback

    def register_abort_grid(self, callback: Callable[[], None]) -> None:
        """Register callback to abort active grid scan."""
        self._abort_grid_callback = callback

    def update_home(self, lat: float, lon: float) -> None:
        """Update the home coordinates. Used when base station moves."""
        with self._lock:
            self._home_lat = lat
            self._home_lon = lon
        logger.info("Home position updated: (%.7f, %.7f)", lat, lon)

    def initiate(self, reason: RTBReason = RTBReason.MANUAL) -> bool:
        """
        Initiate Return To Base.
        Aborts any active navigation/mission and begins driving home.
        Returns True if RTB was initiated, False if already active.
        """
        with self._lock:
            if self._active:
                logger.warning("RTB already active — ignoring duplicate request")
                return False

            self._active = True
            self._state = RTBState.INITIATED
            self._reason = reason
            self._attempts = 0
            self._start_time = time.time()

        logger.warning("RTB INITIATED — reason: %s", reason.value)

        self._abort_active_operations()
        self._publish_rtb_alert("initiated")

        if self._on_rtb_started:
            try:
                self._on_rtb_started(reason)
            except Exception as e:
                logger.error("RTB started callback error: %s", e)

        self._log_event("rtb_initiated", f"RTB initiated: {reason.value}")

        self._running = True
        self._thread = threading.Thread(
            target=self._rtb_loop,
            name="rtb-executor",
            daemon=True
        )
        self._thread.start()

        return True

    def cancel(self) -> None:
        """Cancel an active RTB. Only use if conditions have changed."""
        if not self._active:
            return

        self._running = False
        self._waypoint_arrived.set()
        self._navigator.stop_navigation()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        with self._lock:
            self._active = False
            self._state = RTBState.IDLE

        logger.info("RTB cancelled")

    def _abort_active_operations(self) -> None:
        """Abort any active mission, patrol, or grid scan."""
        self._navigator.stop_navigation()
        self._drive.stop()

        if self._abort_mission_callback:
            try:
                self._abort_mission_callback()
            except Exception as e:
                logger.error("Mission abort callback error: %s", e)

        if self._abort_patrol_callback:
            try:
                self._abort_patrol_callback()
            except Exception as e:
                logger.error("Patrol abort callback error: %s", e)

        if self._abort_grid_callback:
            try:
                self._abort_grid_callback()
            except Exception as e:
                logger.error("Grid abort callback error: %s", e)

        logger.info("Active operations aborted for RTB")

    def _rtb_loop(self) -> None:
        """Main RTB execution loop with retry logic."""
        logger.info("RTB loop started — heading home")

        while self._running and self._attempts < self._max_retries:
            with self._lock:
                self._attempts += 1

            logger.info("RTB attempt %d/%d", self._attempts, self._max_retries)

            if not self._wait_for_gps():
                if self._attempts < self._max_retries:
                    logger.warning("No GPS fix — waiting %.0fs before retry", self._retry_delay)
                    self._interruptible_sleep(self._retry_delay)
                    continue
                else:
                    break

            success: bool = self._navigate_home()

            if success:
                self._handle_arrival()
                return

            if self._attempts < self._max_retries and self._running:
                logger.warning("RTB navigation failed — retrying in %.0fs", self._retry_delay)
                self._interruptible_sleep(self._retry_delay)

        if self._running:
            self._handle_failure()

    def _wait_for_gps(self) -> bool:
        """Wait for a GPS fix before attempting navigation."""
        pos: tuple[float, float] = self._navigator.current_position
        if pos[0] != 0.0 or pos[1] != 0.0:
            return True

        logger.info("Waiting for GPS fix (timeout=%.0fs)...", self._gps_wait_timeout)
        end_time: float = time.time() + self._gps_wait_timeout

        while time.time() < end_time and self._running:
            pos = self._navigator.current_position
            if pos[0] != 0.0 or pos[1] != 0.0:
                logger.info("GPS fix acquired for RTB")
                return True
            time.sleep(1.0)

        logger.warning("GPS fix timeout during RTB")
        return False

    def _navigate_home(self) -> bool:
        """Navigate to the home waypoint."""
        with self._lock:
            self._state = RTBState.NAVIGATING
            speed: float = self._cruise_speed

            if self._reason in (RTBReason.CRITICAL_BATTERY, RTBReason.THERMAL):
                speed = self._approach_speed

        home_wp: Waypoint = Waypoint(
            lat=self._home_lat,
            lon=self._home_lon,
            name="home_base",
            radius_m=self._home_radius,
            speed=speed
        )

        self._waypoint_arrived.clear()
        self._navigator.set_waypoints([home_wp])

        self._navigator.register_waypoint_callback(
            lambda idx, w: self._waypoint_arrived.set()
        )

        if not self._navigator.start_navigation():
            logger.error("Failed to start RTB navigation")
            return False

        self._publish_rtb_alert("navigating")

        while not self._waypoint_arrived.is_set() and self._running:
            nav_state: NavState = self._navigator.nav_state

            if nav_state == NavState.ERROR:
                logger.error("Navigation error during RTB")
                self._navigator.stop_navigation()
                return False

            if nav_state == NavState.NO_FIX:
                logger.warning("GPS fix lost during RTB — waiting")

            status = self._navigator.status
            if status.distance_m > 0 and status.distance_m < self._home_radius * 5:
                with self._lock:
                    self._state = RTBState.APPROACHING

            self._waypoint_arrived.wait(timeout=2.0)

        self._navigator.stop_navigation()
        return self._waypoint_arrived.is_set()

    def _handle_arrival(self) -> None:
        """Handle successful arrival at home base."""
        self._drive.stop()

        with self._lock:
            self._state = RTBState.ARRIVED
            self._active = False
            elapsed: float = time.time() - self._start_time

        self._running = False

        logger.info(
            "RTB ARRIVED — home reached in %.1fs (%d attempts), reason: %s",
            elapsed, self._attempts, self._reason.value
        )

        self._log_event(
            "rtb_arrived",
            f"RTB arrived: {self._reason.value}, {elapsed:.1f}s, {self._attempts} attempts"
        )
        self._publish_rtb_alert("arrived")

        if self._on_rtb_arrived:
            try:
                self._on_rtb_arrived()
            except Exception as e:
                logger.error("RTB arrived callback error: %s", e)

    def _handle_failure(self) -> None:
        """Handle RTB failure — safe stop in place."""
        self._drive.emergency_stop()

        with self._lock:
            self._state = RTBState.SAFE_STOP
            self._active = False
            elapsed: float = time.time() - self._start_time

        self._running = False

        failure_msg: str = (
            f"RTB FAILED — could not reach home after {self._attempts} attempts "
            f"({elapsed:.1f}s). Performing safe stop in place."
        )

        logger.critical(failure_msg)

        self._log_event("rtb_failed", failure_msg)
        self._publish_rtb_alert("failed")

        if self._on_rtb_failed:
            try:
                self._on_rtb_failed(failure_msg)
            except Exception as e:
                logger.error("RTB failed callback error: %s", e)

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that can be interrupted by stop/cancel."""
        end_time: float = time.time() + seconds
        while time.time() < end_time and self._running:
            time.sleep(0.5)

    def _publish_rtb_alert(self, event: str) -> None:
        """Publish RTB alert to MQTT."""
        if self._mqtt is None:
            return
        try:
            self._mqtt.publish_alert({
                "type": f"rtb_{event}",
                "reason": self._reason.value,
                "state": self._state.value,
                "home_lat": self._home_lat,
                "home_lon": self._home_lon,
                "attempts": self._attempts,
                "elapsed_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0
            })
        except Exception:
            pass

    def _log_event(self, event_type: str, message: str) -> None:
        """Log RTB event to database."""
        if self._db is None:
            return
        try:
            self._db.log_mission_event(
                mission_name="rtb",
                event_type=event_type,
                message=message
            )
        except Exception as e:
            logger.error("Failed to log RTB event: %s", e)

    def handle_safety_rtb(self, violation: Any) -> None:
        """
        Safety watchdog callback — triggers RTB from safety system.
        Registered with SafetyWatchdog for FORCE_RTB action.
        """
        reason: RTBReason = RTBReason.SAFETY_VIOLATION

        source: str = getattr(violation, "source", "")
        if source == "battery":
            message: str = getattr(violation, "message", "")
            if "critical" in message.lower():
                reason = RTBReason.CRITICAL_BATTERY
            else:
                reason = RTBReason.LOW_BATTERY
        elif source == "thermal":
            reason = RTBReason.THERMAL

        self.initiate(reason)

    def handle_remote_rtb(self, message: Any) -> None:
        """
        MQTT command callback — triggers RTB from Dashboard.
        Registered with CerberusMQTT for cerberus/command/rtb topic.
        """
        self.initiate(RTBReason.REMOTE_COMMAND)

    @property
    def status(self) -> RTBStatus:
        """Get current RTB status."""
        with self._lock:
            elapsed: float = 0.0
            if self._start_time > 0 and self._active:
                elapsed = time.time() - self._start_time

            distance: float = 0.0
            if self._active:
                nav_status = self._navigator.status
                distance = nav_status.distance_m

            return RTBStatus(
                state=self._state,
                reason=self._reason,
                home_lat=self._home_lat,
                home_lon=self._home_lon,
                distance_to_home_m=distance,
                elapsed_seconds=elapsed,
                attempts=self._attempts
            )

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def rtb_state(self) -> RTBState:
        with self._lock:
            return self._state

    @property
    def home_position(self) -> tuple[float, float]:
        return (self._home_lat, self._home_lon)

    def __repr__(self) -> str:
        return (
            f"ReturnToBase(state={self._state.value}, "
            f"active={self._active}, "
            f"reason={self._reason.value}, "
            f"attempts={self._attempts})"
        )