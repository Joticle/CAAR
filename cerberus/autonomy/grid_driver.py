"""
Cerberus Grid Pattern Driver
Drives systematic grid patterns over a defined area for complete
coverage. Used by weed scanner and microclimate mapper to ensure
every square meter is surveyed. Generates a boustrophedon (lawn mower)
path — back and forth rows with turns at each end.
"""

import math
import time
import logging
import threading
from typing import Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig
from cerberus.mobility.navigator import Navigator, Waypoint


logger: logging.Logger = logging.getLogger(__name__)

EARTH_RADIUS_M: float = 6371000.0


class GridState(Enum):
    """Grid execution states."""
    IDLE = "idle"
    GENERATING = "generating"
    RUNNING = "running"
    AT_POINT = "at_point"
    EXECUTING_ACTION = "executing_action"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class GridPoint:
    """A single point in the grid pattern."""
    lat: float = 0.0
    lon: float = 0.0
    row: int = 0
    col: int = 0
    index: int = 0
    visited: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "row": self.row,
            "col": self.col,
            "index": self.index,
            "visited": self.visited,
            "data": self.data
        }


@dataclass
class GridDefinition:
    """Defines a rectangular grid area for systematic coverage."""
    name: str = ""
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    width_m: float = 10.0
    height_m: float = 10.0
    spacing_m: float = 1.0
    heading_deg: float = 0.0
    speed: float = 0.3
    dwell_seconds: float = 3.0
    action: str = "scan"
    waypoint_radius_m: float = 1.0
    max_duration_minutes: float = 30.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin_lat": round(self.origin_lat, 7),
            "origin_lon": round(self.origin_lon, 7),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "spacing_m": self.spacing_m,
            "heading_deg": self.heading_deg,
            "speed": self.speed,
            "dwell_seconds": self.dwell_seconds,
            "action": self.action,
            "max_duration_minutes": self.max_duration_minutes
        }


@dataclass
class GridStatus:
    """Current grid execution status."""
    state: GridState = GridState.IDLE
    grid_name: str = ""
    total_points: int = 0
    points_visited: int = 0
    rows: int = 0
    cols: int = 0
    current_row: int = 0
    current_col: int = 0
    coverage_pct: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "grid_name": self.grid_name,
            "total_points": self.total_points,
            "points_visited": self.points_visited,
            "rows": self.rows,
            "cols": self.cols,
            "current_row": self.current_row,
            "current_col": self.current_col,
            "coverage_pct": round(self.coverage_pct, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1)
        }


class GridDriver:
    """
    Generates and executes boustrophedon (lawn mower) grid patterns.
    Given a rectangular area defined by origin, width, height, and spacing,
    generates a grid of GPS waypoints and drives through them systematically.

    Pattern:
        Row 0: left → right
        Row 1: right → left (reversed)
        Row 2: left → right
        ... alternating back and forth

    The grid can be rotated by heading_deg to align with property
    boundaries or terrain features.
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

        self._default_spacing: float = config.get("mission", "grid_spacing_m", default=2.0)
        self._default_speed: float = config.get("mission", "grid_speed", default=0.3)

        self._grid_def: Optional[GridDefinition] = None
        self._points: list[GridPoint] = []
        self._execution_order: list[int] = []
        self._rows: int = 0
        self._cols: int = 0

        self._state: GridState = GridState.IDLE
        self._current_point_index: int = 0
        self._points_visited: int = 0
        self._start_time: float = 0.0

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()
        self._waypoint_arrived: threading.Event = threading.Event()

        self._action_handler: Optional[Callable[[GridPoint], dict[str, Any]]] = None
        self._on_point_visited: Optional[Callable[[GridPoint], None]] = None
        self._on_grid_complete: Optional[Callable[[list[GridPoint]], None]] = None

        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None
        self._safety: Optional[Any] = None

        logger.info("Grid driver created")

    def bind_db(self, db: Any) -> None:
        """Bind database for grid logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for status publishing."""
        self._mqtt = mqtt_client

    def bind_safety(self, safety: Any) -> None:
        """Bind safety watchdog."""
        self._safety = safety

    def register_action_handler(
        self,
        handler: Callable[[GridPoint], dict[str, Any]]
    ) -> None:
        """
        Register the action handler for each grid point.
        Handler receives a GridPoint, performs the action (sensor read,
        photo capture, etc), and returns data dict to store on the point.
        """
        self._action_handler = handler

    def register_point_callback(
        self,
        callback: Callable[[GridPoint], None]
    ) -> None:
        """Register callback for each point visit: callback(point)."""
        self._on_point_visited = callback

    def register_complete_callback(
        self,
        callback: Callable[[list[GridPoint]], None]
    ) -> None:
        """Register callback for grid completion: callback(all_points)."""
        self._on_grid_complete = callback

    def define_grid(self, grid_def: GridDefinition) -> bool:
        """Define and generate a grid pattern."""
        if grid_def.width_m <= 0 or grid_def.height_m <= 0:
            logger.error("Invalid grid dimensions: %.1f x %.1f", grid_def.width_m, grid_def.height_m)
            return False

        if grid_def.spacing_m <= 0:
            logger.error("Invalid grid spacing: %.1f", grid_def.spacing_m)
            return False

        with self._lock:
            self._state = GridState.GENERATING
            self._grid_def = grid_def

        self._generate_grid(grid_def)
        self._generate_execution_order()

        with self._lock:
            self._current_point_index = 0
            self._points_visited = 0
            self._state = GridState.IDLE

        logger.info(
            "Grid generated: '%s' — %d rows x %d cols = %d points, spacing=%.1fm",
            grid_def.name, self._rows, self._cols, len(self._points), grid_def.spacing_m
        )
        return True

    def define_grid_from_task(self, task: Any) -> bool:
        """Define a grid from a mission task's parameters."""
        params: dict[str, Any] = task.parameters

        grid_def: GridDefinition = GridDefinition(
            name=params.get("grid_name", task.name),
            origin_lat=params.get("origin_lat", task.waypoint_lat),
            origin_lon=params.get("origin_lon", task.waypoint_lon),
            width_m=params.get("width_m", 10.0),
            height_m=params.get("height_m", 10.0),
            spacing_m=params.get("spacing_m", self._default_spacing),
            heading_deg=params.get("heading_deg", 0.0),
            speed=params.get("speed", self._default_speed),
            dwell_seconds=params.get("dwell_seconds", 3.0),
            action=params.get("action", "scan"),
            waypoint_radius_m=params.get("radius_m", 1.0),
            max_duration_minutes=params.get("max_duration_minutes", 30.0),
            parameters=params
        )

        return self.define_grid(grid_def)

    def _generate_grid(self, grid_def: GridDefinition) -> None:
        """Generate GPS coordinates for all grid points."""
        self._cols = max(1, int(math.ceil(grid_def.width_m / grid_def.spacing_m)) + 1)
        self._rows = max(1, int(math.ceil(grid_def.height_m / grid_def.spacing_m)) + 1)
        self._points = []

        heading_rad: float = math.radians(grid_def.heading_deg)
        cos_h: float = math.cos(heading_rad)
        sin_h: float = math.sin(heading_rad)

        index: int = 0
        for row in range(self._rows):
            for col in range(self._cols):
                local_x: float = col * grid_def.spacing_m
                local_y: float = row * grid_def.spacing_m

                rotated_x: float = local_x * cos_h - local_y * sin_h
                rotated_y: float = local_x * sin_h + local_y * cos_h

                lat, lon = self._offset_gps(
                    grid_def.origin_lat,
                    grid_def.origin_lon,
                    rotated_y,
                    rotated_x
                )

                point: GridPoint = GridPoint(
                    lat=lat,
                    lon=lon,
                    row=row,
                    col=col,
                    index=index
                )
                self._points.append(point)
                index += 1

    def _generate_execution_order(self) -> None:
        """Generate boustrophedon (lawn mower) execution order."""
        self._execution_order = []

        for row in range(self._rows):
            row_start: int = row * self._cols
            row_indices: list[int] = list(range(row_start, row_start + self._cols))

            if row % 2 == 1:
                row_indices.reverse()

            self._execution_order.extend(row_indices)

    @staticmethod
    def _offset_gps(
        lat: float,
        lon: float,
        north_m: float,
        east_m: float
    ) -> tuple[float, float]:
        """Offset a GPS coordinate by meters north and east."""
        d_lat: float = north_m / EARTH_RADIUS_M
        d_lon: float = east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))

        new_lat: float = lat + math.degrees(d_lat)
        new_lon: float = lon + math.degrees(d_lon)

        return (new_lat, new_lon)

    def start(self) -> bool:
        """Start grid execution."""
        with self._lock:
            if not self._points:
                logger.error("No grid defined")
                return False
            if self._running:
                logger.warning("Grid already running")
                return False

        if self._safety is not None and not self._safety.is_safe_for_mission:
            logger.warning("Cannot start grid — safety violation active")
            return False

        self._running = True
        self._start_time = time.time()

        with self._lock:
            self._state = GridState.RUNNING

        self._thread = threading.Thread(
            target=self._grid_loop,
            name="grid-driver",
            daemon=True
        )
        self._thread.start()

        logger.info("Grid scan STARTED: '%s'", self._grid_def.name if self._grid_def else "unnamed")
        return True

    def pause(self) -> None:
        """Pause grid execution."""
        with self._lock:
            if self._state != GridState.RUNNING:
                return
            self._state = GridState.PAUSED

        self._pause_event.clear()
        self._navigator.pause()
        logger.info("Grid scan PAUSED at point %d/%d", self._points_visited, len(self._points))

    def resume(self) -> None:
        """Resume grid execution."""
        with self._lock:
            if self._state != GridState.PAUSED:
                return
            self._state = GridState.RUNNING

        self._pause_event.set()
        self._navigator.resume()
        logger.info("Grid scan RESUMED")

    def stop_grid(self) -> None:
        """Stop grid execution."""
        if not self._running:
            return

        self._running = False
        self._pause_event.set()
        self._waypoint_arrived.set()
        self._navigator.stop_navigation()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Grid thread did not stop cleanly")
            self._thread = None

        with self._lock:
            self._state = GridState.IDLE

        logger.info("Grid scan STOPPED")

    def _grid_loop(self) -> None:
        """Main grid execution loop."""
        logger.info("Grid loop started — %d points to visit", len(self._execution_order))

        while self._running and self._current_point_index < len(self._execution_order):
            self._pause_event.wait()
            if not self._running:
                break

            if self._safety is not None and not self._safety.is_safe_for_mission:
                logger.warning("Safety violation during grid scan — stopping")
                self._running = False
                with self._lock:
                    self._state = GridState.ERROR
                break

            if self._grid_def:
                elapsed_min: float = (time.time() - self._start_time) / 60.0
                if elapsed_min > self._grid_def.max_duration_minutes:
                    logger.warning("Grid time limit exceeded (%.1f min)", elapsed_min)
                    break

            point_idx: int = self._execution_order[self._current_point_index]
            point: GridPoint = self._points[point_idx]

            success: bool = self._navigate_to_point(point)

            if not self._running:
                break

            if success:
                self._execute_point_arrival(point)
            else:
                logger.warning(
                    "Failed to reach grid point [%d,%d] — skipping",
                    point.row, point.col
                )

            with self._lock:
                self._current_point_index += 1

        self._complete_grid()

    def _navigate_to_point(self, point: GridPoint) -> bool:
        """Navigate to a single grid point."""
        with self._lock:
            self._state = GridState.RUNNING

        radius: float = self._grid_def.waypoint_radius_m if self._grid_def else 1.0
        speed: float = self._grid_def.speed if self._grid_def else 0.3

        wp: Waypoint = Waypoint(
            lat=point.lat,
            lon=point.lon,
            name=f"grid_{point.row}_{point.col}",
            radius_m=radius,
            speed=speed
        )

        self._waypoint_arrived.clear()
        self._navigator.set_waypoints([wp])

        self._navigator.register_waypoint_callback(
            lambda idx, w: self._waypoint_arrived.set()
        )

        if not self._navigator.start_navigation():
            return False

        while not self._waypoint_arrived.is_set() and self._running:
            self._waypoint_arrived.wait(timeout=1.0)

        self._navigator.stop_navigation()
        return self._waypoint_arrived.is_set()

    def _execute_point_arrival(self, point: GridPoint) -> None:
        """Execute actions at a grid point."""
        with self._lock:
            self._state = GridState.AT_POINT
            self._points_visited += 1
            point.visited = True

        logger.debug(
            "Grid point [%d,%d] reached — %d/%d (%.1f%%)",
            point.row, point.col,
            self._points_visited, len(self._points),
            (self._points_visited / len(self._points)) * 100
        )

        if self._action_handler is not None:
            with self._lock:
                self._state = GridState.EXECUTING_ACTION
            try:
                result: dict[str, Any] = self._action_handler(point)
                point.data = result
            except Exception as e:
                logger.error("Grid action error at [%d,%d]: %s", point.row, point.col, e)

        dwell: float = self._grid_def.dwell_seconds if self._grid_def else 0.0
        if dwell > 0:
            end_time: float = time.time() + dwell
            while time.time() < end_time and self._running:
                self._pause_event.wait(timeout=0.5)
                if not self._running:
                    return

        if self._on_point_visited:
            try:
                self._on_point_visited(point)
            except Exception as e:
                logger.error("Point visited callback error: %s", e)

    def _complete_grid(self) -> None:
        """Handle grid completion."""
        self._running = False

        with self._lock:
            if self._state != GridState.ERROR:
                self._state = GridState.COMPLETED

        if self._grid_def:
            elapsed: float = time.time() - self._start_time
            coverage: float = (self._points_visited / len(self._points) * 100) if self._points else 0

            logger.info(
                "Grid '%s' COMPLETED — %d/%d points (%.1f%% coverage) in %.1fs",
                self._grid_def.name, self._points_visited, len(self._points),
                coverage, elapsed
            )

            if self._db is not None:
                try:
                    self._db.log_mission_event(
                        mission_name=self._grid_def.name,
                        event_type="grid_complete",
                        message=f"Grid complete: {self._points_visited}/{len(self._points)} points, {coverage:.1f}%"
                    )
                except Exception as e:
                    logger.error("Failed to log grid completion: %s", e)

        if self._on_grid_complete:
            try:
                self._on_grid_complete(self._points)
            except Exception as e:
                logger.error("Grid complete callback error: %s", e)

    def execute_mission_task(self, task: Any) -> bool:
        """
        Execute a grid scan task from the mission planner.
        This is the handler registered with MissionPlanner for GRID_SCAN tasks.
        """
        if not self.define_grid_from_task(task):
            return False

        if not self.start():
            return False

        while self._running:
            time.sleep(1.0)

        with self._lock:
            return self._state == GridState.COMPLETED

    @property
    def points(self) -> list[GridPoint]:
        """Get all grid points with visit status and collected data."""
        return list(self._points)

    @property
    def visited_points(self) -> list[GridPoint]:
        """Get only visited grid points."""
        return [p for p in self._points if p.visited]

    @property
    def status(self) -> GridStatus:
        """Get current grid status."""
        with self._lock:
            elapsed: float = 0.0
            if self._start_time > 0 and self._running:
                elapsed = time.time() - self._start_time

            coverage: float = 0.0
            if self._points:
                coverage = (self._points_visited / len(self._points)) * 100

            current_row: int = 0
            current_col: int = 0
            if self._current_point_index < len(self._execution_order):
                idx: int = self._execution_order[self._current_point_index]
                if idx < len(self._points):
                    current_row = self._points[idx].row
                    current_col = self._points[idx].col

            return GridStatus(
                state=self._state,
                grid_name=self._grid_def.name if self._grid_def else "",
                total_points=len(self._points),
                points_visited=self._points_visited,
                rows=self._rows,
                cols=self._cols,
                current_row=current_row,
                current_col=current_col,
                coverage_pct=coverage,
                elapsed_seconds=elapsed
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def grid_state(self) -> GridState:
        with self._lock:
            return self._state

    def __repr__(self) -> str:
        return (
            f"GridDriver(state={self._state.value}, "
            f"grid='{self._grid_def.name if self._grid_def else 'none'}', "
            f"visited={self._points_visited}/{len(self._points)})"
        )