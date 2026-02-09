"""
Cerberus Autonomous Patrol
Executes predefined patrol routes — sequences of waypoints that
Cerberus drives autonomously while performing head-specific actions
at each stop. Supports looping, randomization, and adaptive timing.
"""

import time
import random
import logging
import threading
from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig
from cerberus.mobility.navigator import Navigator, Waypoint, NavState
from cerberus.autonomy.mission import MissionTask


logger: logging.Logger = logging.getLogger(__name__)


class PatrolState(Enum):
    """Patrol execution states."""
    IDLE = "idle"
    PATROLLING = "patrolling"
    AT_WAYPOINT = "at_waypoint"
    EXECUTING_ACTION = "executing_action"
    RETURNING = "returning"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PatrolWaypoint:
    """A waypoint in a patrol route with associated actions."""
    lat: float = 0.0
    lon: float = 0.0
    name: str = ""
    radius_m: float = 2.0
    speed: float = 0.5
    dwell_seconds: float = 5.0
    action: str = "scan"
    heading_deg: Optional[float] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_waypoint(self) -> Waypoint:
        """Convert to Navigator Waypoint."""
        return Waypoint(
            lat=self.lat,
            lon=self.lon,
            name=self.name,
            radius_m=self.radius_m,
            speed=self.speed,
            action=self.action,
            metadata=self.parameters
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "name": self.name,
            "radius_m": self.radius_m,
            "speed": self.speed,
            "dwell_seconds": self.dwell_seconds,
            "action": self.action,
            "heading_deg": self.heading_deg,
            "parameters": self.parameters
        }


@dataclass
class PatrolRoute:
    """A complete patrol route definition."""
    name: str = ""
    description: str = ""
    waypoints: list[PatrolWaypoint] = field(default_factory=list)
    loop: bool = False
    max_loops: int = 1
    randomize_order: bool = False
    randomize_dwell: bool = False
    dwell_variance_pct: float = 25.0
    reverse_alternate: bool = False
    max_duration_minutes: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "waypoint_count": len(self.waypoints),
            "loop": self.loop,
            "max_loops": self.max_loops,
            "randomize_order": self.randomize_order,
            "max_duration_minutes": self.max_duration_minutes
        }


@dataclass
class PatrolStatus:
    """Current patrol execution status."""
    state: PatrolState = PatrolState.IDLE
    route_name: str = ""
    current_waypoint_index: int = 0
    total_waypoints: int = 0
    waypoints_visited: int = 0
    current_loop: int = 0
    max_loops: int = 1
    elapsed_seconds: float = 0.0
    current_waypoint_name: str = ""
    current_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "route_name": self.route_name,
            "current_waypoint_index": self.current_waypoint_index,
            "total_waypoints": self.total_waypoints,
            "waypoints_visited": self.waypoints_visited,
            "current_loop": self.current_loop,
            "max_loops": self.max_loops,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "current_waypoint_name": self.current_waypoint_name,
            "current_action": self.current_action
        }


class PatrolExecutor:
    """
    Executes patrol routes autonomously.
    Drives between waypoints using the Navigator, executes actions
    at each stop, and supports looping, randomization, and
    adaptive dwell times for unpredictable patrol patterns.
    """

    def __init__(
        self,
        navigator: Navigator,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._navigator: Navigator = navigator
        self._config: CerberusConfig = config

        self._default_dwell: float = config.get("mission", "patrol_dwell_seconds", default=10.0)
        self._default_speed: float = config.get("mission", "patrol_speed", default=0.4)

        self._route: Optional[PatrolRoute] = None
        self._execution_order: list[int] = []
        self._state: PatrolState = PatrolState.IDLE
        self._current_wp_index: int = 0
        self._current_loop: int = 0
        self._waypoints_visited: int = 0
        self._start_time: float = 0.0

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()
        self._waypoint_arrived: threading.Event = threading.Event()

        self._action_handlers: dict[str, Callable[[PatrolWaypoint], bool]] = {}
        self._on_waypoint_visited: Optional[Callable[[int, PatrolWaypoint], None]] = None
        self._on_patrol_complete: Optional[Callable[[PatrolRoute], None]] = None
        self._on_loop_complete: Optional[Callable[[int], None]] = None

        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None
        self._safety: Optional[Any] = None

        logger.info("Patrol executor created")

    def bind_db(self, db: Any) -> None:
        """Bind database for patrol logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for status publishing."""
        self._mqtt = mqtt_client

    def bind_safety(self, safety: Any) -> None:
        """Bind safety watchdog for patrol checks."""
        self._safety = safety

    def register_action_handler(
        self,
        action: str,
        handler: Callable[[PatrolWaypoint], bool]
    ) -> None:
        """
        Register a handler for a waypoint action.
        The active head registers its scan/capture/detect actions.
        handler(waypoint) returns True on success.
        """
        self._action_handlers[action] = handler
        logger.info("Patrol action handler registered: %s", action)

    def register_waypoint_callback(
        self,
        callback: Callable[[int, PatrolWaypoint], None]
    ) -> None:
        """Register callback for waypoint visits: callback(index, waypoint)."""
        self._on_waypoint_visited = callback

    def register_patrol_complete_callback(
        self,
        callback: Callable[[PatrolRoute], None]
    ) -> None:
        """Register callback for patrol completion: callback(route)."""
        self._on_patrol_complete = callback

    def register_loop_complete_callback(
        self,
        callback: Callable[[int], None]
    ) -> None:
        """Register callback for loop completion: callback(loop_number)."""
        self._on_loop_complete = callback

    def load_route(self, route: PatrolRoute) -> bool:
        """Load a patrol route for execution."""
        if not route.waypoints:
            logger.error("Cannot load empty patrol route")
            return False

        with self._lock:
            self._route = route
            self._current_wp_index = 0
            self._current_loop = 0
            self._waypoints_visited = 0
            self._execution_order = list(range(len(route.waypoints)))
            self._state = PatrolState.IDLE

        logger.info(
            "Patrol route loaded: '%s' — %d waypoints, loop=%s, max_loops=%d",
            route.name, len(route.waypoints), route.loop, route.max_loops
        )
        return True

    def load_route_from_task(self, task: MissionTask) -> bool:
        """Load a patrol route from a mission task's parameters."""
        params: dict[str, Any] = task.parameters
        raw_waypoints: list[dict] = params.get("waypoints", [])

        if not raw_waypoints:
            if task.waypoint_lat != 0.0 or task.waypoint_lon != 0.0:
                raw_waypoints = [{
                    "lat": task.waypoint_lat,
                    "lon": task.waypoint_lon,
                    "name": task.name,
                    "speed": task.speed,
                    "dwell": task.duration_seconds
                }]

        waypoints: list[PatrolWaypoint] = []
        for i, wp_data in enumerate(raw_waypoints):
            waypoints.append(PatrolWaypoint(
                lat=wp_data.get("lat", 0.0),
                lon=wp_data.get("lon", 0.0),
                name=wp_data.get("name", f"patrol_{i}"),
                radius_m=wp_data.get("radius_m", 2.0),
                speed=wp_data.get("speed", self._default_speed),
                dwell_seconds=wp_data.get("dwell", self._default_dwell),
                action=wp_data.get("action", "scan"),
                heading_deg=wp_data.get("heading", None),
                parameters=wp_data.get("parameters", {})
            ))

        route: PatrolRoute = PatrolRoute(
            name=params.get("route_name", task.name),
            description=params.get("description", ""),
            waypoints=waypoints,
            loop=params.get("loop", False),
            max_loops=params.get("max_loops", 1),
            randomize_order=params.get("randomize_order", False),
            randomize_dwell=params.get("randomize_dwell", False),
            dwell_variance_pct=params.get("dwell_variance_pct", 25.0),
            reverse_alternate=params.get("reverse_alternate", False),
            max_duration_minutes=params.get("max_duration_minutes", 30.0)
        )

        return self.load_route(route)

    def start(self) -> bool:
        """Start patrol execution."""
        with self._lock:
            if self._route is None:
                logger.error("No patrol route loaded")
                return False
            if self._running:
                logger.warning("Patrol already running")
                return False

        if self._safety is not None and not self._safety.is_safe_for_mission:
            logger.warning("Cannot start patrol — safety violation active")
            return False

        self._running = True
        self._start_time = time.time()

        with self._lock:
            self._state = PatrolState.PATROLLING

        if self._route.randomize_order:
            self._shuffle_order()

        self._thread = threading.Thread(
            target=self._patrol_loop,
            name="patrol-executor",
            daemon=True
        )
        self._thread.start()

        logger.info("Patrol STARTED: '%s'", self._route.name)
        self._log_event("patrol_start", f"Patrol '{self._route.name}' started")

        return True

    def pause(self) -> None:
        """Pause the patrol."""
        with self._lock:
            if self._state != PatrolState.PATROLLING:
                return
            self._state = PatrolState.PAUSED

        self._pause_event.clear()
        self._navigator.pause()
        logger.info("Patrol PAUSED")

    def resume(self) -> None:
        """Resume a paused patrol."""
        with self._lock:
            if self._state != PatrolState.PAUSED:
                return
            self._state = PatrolState.PATROLLING

        self._pause_event.set()
        self._navigator.resume()
        logger.info("Patrol RESUMED")

    def stop_patrol(self) -> None:
        """Stop patrol execution."""
        if not self._running:
            return

        self._running = False
        self._pause_event.set()
        self._waypoint_arrived.set()
        self._navigator.stop_navigation()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Patrol thread did not stop cleanly")
            self._thread = None

        with self._lock:
            self._state = PatrolState.IDLE

        logger.info("Patrol STOPPED")

    def _patrol_loop(self) -> None:
        """Main patrol execution loop."""
        logger.info("Patrol loop started")

        while self._running:
            self._pause_event.wait()
            if not self._running:
                break

            if self._safety is not None and not self._safety.is_safe_for_mission:
                logger.warning("Safety violation during patrol — stopping")
                self._running = False
                with self._lock:
                    self._state = PatrolState.ERROR
                break

            with self._lock:
                if self._route is None:
                    break

                elapsed_min: float = (time.time() - self._start_time) / 60.0
                if elapsed_min > self._route.max_duration_minutes:
                    logger.warning("Patrol time limit exceeded (%.1f min)", elapsed_min)
                    break

                if self._current_wp_index >= len(self._execution_order):
                    if self._handle_loop_end():
                        continue
                    else:
                        break

                wp_idx: int = self._execution_order[self._current_wp_index]
                wp: PatrolWaypoint = self._route.waypoints[wp_idx]

            success: bool = self._navigate_to_waypoint(wp)

            if not self._running:
                break

            if success:
                self._execute_waypoint_arrival(wp, wp_idx)
            else:
                logger.warning("Failed to reach waypoint '%s' — skipping", wp.name)

            with self._lock:
                self._current_wp_index += 1

        self._complete_patrol()

    def _navigate_to_waypoint(self, wp: PatrolWaypoint) -> bool:
        """Navigate to a single patrol waypoint."""
        with self._lock:
            self._state = PatrolState.PATROLLING

        logger.info("Navigating to waypoint: %s (%.7f, %.7f)", wp.name, wp.lat, wp.lon)

        self._waypoint_arrived.clear()
        self._navigator.set_waypoints([wp.to_waypoint()])

        self._navigator.register_waypoint_callback(
            lambda idx, w: self._waypoint_arrived.set()
        )

        if not self._navigator.start_navigation():
            logger.error("Failed to start navigation to %s", wp.name)
            return False

        while not self._waypoint_arrived.is_set() and self._running:
            if self._navigator.nav_state == NavState.ERROR:
                logger.error("Navigation error while heading to %s", wp.name)
                self._navigator.stop_navigation()
                return False

            self._waypoint_arrived.wait(timeout=1.0)

        self._navigator.stop_navigation()
        return self._waypoint_arrived.is_set()

    def _execute_waypoint_arrival(self, wp: PatrolWaypoint, wp_idx: int) -> None:
        """Execute actions when arriving at a patrol waypoint."""
        with self._lock:
            self._state = PatrolState.AT_WAYPOINT
            self._waypoints_visited += 1

        logger.info(
            "Arrived at waypoint: %s — dwell=%.1fs, action=%s",
            wp.name, wp.dwell_seconds, wp.action
        )

        if self._on_waypoint_visited:
            try:
                self._on_waypoint_visited(wp_idx, wp)
            except Exception as e:
                logger.error("Waypoint visited callback error: %s", e)

        if wp.action != "none":
            with self._lock:
                self._state = PatrolState.EXECUTING_ACTION

            handler: Optional[Callable] = self._action_handlers.get(wp.action)
            if handler is not None:
                try:
                    handler(wp)
                except Exception as e:
                    logger.error("Action '%s' failed at %s: %s", wp.action, wp.name, e)
            else:
                logger.debug("No handler for action '%s' — skipping", wp.action)

        dwell: float = self._calculate_dwell(wp)
        if dwell > 0:
            logger.debug("Dwelling at %s for %.1fs", wp.name, dwell)
            end_time: float = time.time() + dwell
            while time.time() < end_time and self._running:
                self._pause_event.wait(timeout=0.5)
                if not self._running:
                    return

        self._log_event(
            "waypoint_visited",
            f"Visited {wp.name} (action={wp.action})"
        )

    def _calculate_dwell(self, wp: PatrolWaypoint) -> float:
        """Calculate dwell time, optionally with randomization."""
        base_dwell: float = wp.dwell_seconds

        if self._route and self._route.randomize_dwell and base_dwell > 0:
            variance: float = base_dwell * (self._route.dwell_variance_pct / 100.0)
            dwell: float = base_dwell + random.uniform(-variance, variance)
            return max(1.0, dwell)

        return base_dwell

    def _shuffle_order(self) -> None:
        """Randomize the waypoint execution order."""
        self._execution_order = list(range(len(self._route.waypoints)))
        random.shuffle(self._execution_order)
        logger.info("Patrol order randomized: %s", self._execution_order)

    def _handle_loop_end(self) -> bool:
        """Handle end of a patrol loop. Returns True if looping continues."""
        if not self._route.loop:
            return False

        self._current_loop += 1
        if self._current_loop >= self._route.max_loops:
            logger.info("Patrol max loops reached (%d)", self._route.max_loops)
            return False

        self._current_wp_index = 0

        if self._route.randomize_order:
            self._shuffle_order()
        elif self._route.reverse_alternate and self._current_loop % 2 == 1:
            self._execution_order.reverse()
            logger.info("Patrol loop %d — reversed order", self._current_loop + 1)

        logger.info("Patrol loop %d/%d starting", self._current_loop + 1, self._route.max_loops)

        if self._on_loop_complete:
            try:
                self._on_loop_complete(self._current_loop)
            except Exception as e:
                logger.error("Loop complete callback error: %s", e)

        return True

    def _complete_patrol(self) -> None:
        """Handle patrol completion."""
        self._running = False

        with self._lock:
            if self._state != PatrolState.ERROR:
                self._state = PatrolState.COMPLETED

        if self._route:
            elapsed: float = time.time() - self._start_time
            logger.info(
                "Patrol '%s' COMPLETED — %d waypoints visited, %d loops, %.1fs",
                self._route.name, self._waypoints_visited,
                self._current_loop + 1, elapsed
            )
            self._log_event(
                "patrol_complete",
                f"Patrol '{self._route.name}' completed: {self._waypoints_visited} waypoints"
            )

            if self._on_patrol_complete:
                try:
                    self._on_patrol_complete(self._route)
                except Exception as e:
                    logger.error("Patrol complete callback error: %s", e)

    def _log_event(self, event_type: str, message: str) -> None:
        """Log a patrol event to database."""
        if self._db is None:
            return
        try:
            self._db.log_mission_event(
                mission_name=self._route.name if self._route else "patrol",
                event_type=event_type,
                message=message
            )
        except Exception as e:
            logger.error("Failed to log patrol event: %s", e)

    def execute_mission_task(self, task: MissionTask) -> bool:
        """
        Execute a patrol task from the mission planner.
        This is the handler registered with MissionPlanner for PATROL tasks.
        """
        if not self.load_route_from_task(task):
            return False

        if not self.start():
            return False

        while self._running:
            time.sleep(1.0)

        with self._lock:
            return self._state == PatrolState.COMPLETED

    @property
    def status(self) -> PatrolStatus:
        """Get current patrol status."""
        with self._lock:
            elapsed: float = 0.0
            if self._start_time > 0 and self._running:
                elapsed = time.time() - self._start_time

            wp_name: str = ""
            action: str = ""
            if self._route and self._current_wp_index < len(self._execution_order):
                idx: int = self._execution_order[self._current_wp_index]
                wp: PatrolWaypoint = self._route.waypoints[idx]
                wp_name = wp.name
                action = wp.action

            return PatrolStatus(
                state=self._state,
                route_name=self._route.name if self._route else "",
                current_waypoint_index=self._current_wp_index,
                total_waypoints=len(self._route.waypoints) if self._route else 0,
                waypoints_visited=self._waypoints_visited,
                current_loop=self._current_loop,
                max_loops=self._route.max_loops if self._route else 1,
                elapsed_seconds=elapsed,
                current_waypoint_name=wp_name,
                current_action=action
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def patrol_state(self) -> PatrolState:
        with self._lock:
            return self._state

    def __repr__(self) -> str:
        return (
            f"PatrolExecutor(state={self._state.value}, "
            f"route='{self._route.name if self._route else 'none'}', "
            f"visited={self._waypoints_visited})"
        )