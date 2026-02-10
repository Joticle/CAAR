"""
Cerberus Path Planner
A* pathfinding on a 2D occupancy grid built from ultrasonic sensor data
and GPS positions. Routes around known obstacles instead of driving
straight lines between waypoints. Grid persists in SQLite across sessions
so Cerberus builds a map of its environment over time.

Grid Coordinate System:
    - Origin: home position (navigation.home_lat, navigation.home_lon)
    - Each cell represents a square of configurable size (default 0.5m)
    - Grid X increases east, Grid Y increases north
    - GPS coordinates are converted to grid coordinates using haversine offset

Cell States:
    - free: Cerberus has driven through this cell or sensors confirmed clear
    - occupied: Ultrasonic sensor detected an obstacle
    - unknown: Never observed
"""

import math
import heapq
import logging
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)

EARTH_RADIUS_M: float = 6371000.0


class CellState(Enum):
    """Occupancy grid cell state."""
    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass
class GridCell:
    """A single cell in the occupancy grid."""
    x: int = 0
    y: int = 0
    state: CellState = CellState.UNKNOWN
    confidence: float = 0.0
    observation_count: int = 0
    last_observed: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0

    @property
    def traversable(self) -> bool:
        return self.state != CellState.OCCUPIED


@dataclass
class PathNode:
    """A node in the A* search with priority ordering."""
    x: int = 0
    y: int = 0
    g_cost: float = 0.0
    h_cost: float = 0.0
    parent: Optional["PathNode"] = field(default=None, repr=False, compare=False)

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: "PathNode") -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class PlannedPath:
    """Result of a path planning operation."""
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    grid_path: list[tuple[int, int]] = field(default_factory=list)
    distance_m: float = 0.0
    cell_count: int = 0
    planning_time_ms: float = 0.0
    success: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoint_count": len(self.waypoints),
            "grid_cells": self.cell_count,
            "distance_m": round(self.distance_m, 2),
            "planning_time_ms": round(self.planning_time_ms, 1),
            "success": self.success,
            "message": self.message
        }


class OccupancyGrid:
    """
    2D occupancy grid centered on the home position. Cells are stored
    in a dict keyed by (x, y) for sparse representation — only observed
    cells consume memory. Unknown cells are implicitly traversable with
    a configurable unknown penalty.
    """

    def __init__(
        self,
        cell_size_m: float = 0.5,
        home_lat: float = 0.0,
        home_lon: float = 0.0
    ) -> None:
        self._cell_size_m: float = cell_size_m
        self._home_lat: float = home_lat
        self._home_lon: float = home_lon
        self._cells: dict[tuple[int, int], GridCell] = {}
        self._obstacle_buffer: int = 1

    @property
    def cell_size_m(self) -> float:
        return self._cell_size_m

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def obstacle_count(self) -> int:
        return sum(1 for c in self._cells.values() if c.state == CellState.OCCUPIED)

    @property
    def free_count(self) -> int:
        return sum(1 for c in self._cells.values() if c.state == CellState.FREE)

    def gps_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert GPS coordinates to grid coordinates."""
        delta_lat: float = lat - self._home_lat
        delta_lon: float = lon - self._home_lon

        meters_north: float = delta_lat * (math.pi / 180.0) * EARTH_RADIUS_M
        meters_east: float = delta_lon * (math.pi / 180.0) * EARTH_RADIUS_M * math.cos(
            math.radians(self._home_lat)
        )

        grid_x: int = int(round(meters_east / self._cell_size_m))
        grid_y: int = int(round(meters_north / self._cell_size_m))

        return grid_x, grid_y

    def grid_to_gps(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid coordinates back to GPS."""
        meters_east: float = grid_x * self._cell_size_m
        meters_north: float = grid_y * self._cell_size_m

        delta_lat: float = meters_north / (EARTH_RADIUS_M * math.pi / 180.0)
        delta_lon: float = meters_east / (
            EARTH_RADIUS_M * math.pi / 180.0 * math.cos(math.radians(self._home_lat))
        )

        return self._home_lat + delta_lat, self._home_lon + delta_lon

    def get_cell(self, x: int, y: int) -> GridCell:
        """Get a cell. Returns unknown cell if not observed."""
        if (x, y) in self._cells:
            return self._cells[(x, y)]
        return GridCell(x=x, y=y, state=CellState.UNKNOWN)

    def set_cell(
        self,
        x: int,
        y: int,
        state: CellState,
        confidence: float = 1.0,
        gps_lat: float = 0.0,
        gps_lon: float = 0.0
    ) -> None:
        """Set or update a cell state."""
        now: float = time.time()

        if (x, y) in self._cells:
            cell: GridCell = self._cells[(x, y)]
            cell.state = state
            cell.confidence = confidence
            cell.observation_count += 1
            cell.last_observed = now
            if gps_lat != 0.0:
                cell.gps_lat = gps_lat
            if gps_lon != 0.0:
                cell.gps_lon = gps_lon
        else:
            self._cells[(x, y)] = GridCell(
                x=x, y=y, state=state, confidence=confidence,
                observation_count=1, last_observed=now,
                gps_lat=gps_lat, gps_lon=gps_lon
            )

    def mark_free(self, lat: float, lon: float) -> None:
        """Mark a GPS position as free (Cerberus drove through it)."""
        x, y = self.gps_to_grid(lat, lon)
        self.set_cell(x, y, CellState.FREE, confidence=1.0, gps_lat=lat, gps_lon=lon)

    def mark_obstacle(
        self,
        lat: float,
        lon: float,
        distance_cm: float,
        heading_deg: float,
        sensor_offset_deg: float = 0.0
    ) -> None:
        """
        Mark an obstacle detected by ultrasonic sensor.
        Calculates the obstacle's GPS position from the rover's position,
        heading, sensor offset angle, and measured distance.
        """
        total_heading: float = (heading_deg + sensor_offset_deg) % 360.0
        heading_rad: float = math.radians(total_heading)
        distance_m: float = distance_cm / 100.0

        obs_meters_east: float = distance_m * math.sin(heading_rad)
        obs_meters_north: float = distance_m * math.cos(heading_rad)

        delta_lat: float = obs_meters_north / (EARTH_RADIUS_M * math.pi / 180.0)
        delta_lon: float = obs_meters_east / (
            EARTH_RADIUS_M * math.pi / 180.0 * math.cos(math.radians(lat))
        )

        obs_lat: float = lat + delta_lat
        obs_lon: float = lon + delta_lon

        x, y = self.gps_to_grid(obs_lat, obs_lon)
        self.set_cell(x, y, CellState.OCCUPIED, confidence=1.0, gps_lat=obs_lat, gps_lon=obs_lon)

        if self._obstacle_buffer > 0:
            for dx in range(-self._obstacle_buffer, self._obstacle_buffer + 1):
                for dy in range(-self._obstacle_buffer, self._obstacle_buffer + 1):
                    if dx == 0 and dy == 0:
                        continue
                    bx: int = x + dx
                    by: int = y + dy
                    existing: GridCell = self.get_cell(bx, by)
                    if existing.state != CellState.OCCUPIED:
                        self.set_cell(bx, by, CellState.OCCUPIED, confidence=0.5)

    def is_traversable(self, x: int, y: int) -> bool:
        """Check if a cell is safe to drive through."""
        cell: GridCell = self.get_cell(x, y)
        return cell.traversable

    def clear_stale_obstacles(self, max_age_seconds: float = 604800.0) -> int:
        """Reset obstacles not observed recently. Default 7 days."""
        now: float = time.time()
        cleared: int = 0
        stale_keys: list[tuple[int, int]] = []

        for key, cell in self._cells.items():
            if cell.state == CellState.OCCUPIED:
                age: float = now - cell.last_observed
                if age > max_age_seconds:
                    stale_keys.append(key)

        for key in stale_keys:
            self._cells[key].state = CellState.UNKNOWN
            self._cells[key].confidence = 0.0
            cleared += 1

        if cleared > 0:
            logger.info("Cleared %d stale obstacles (age > %.0f hours)", cleared, max_age_seconds / 3600)

        return cleared

    def get_all_cells(self) -> list[GridCell]:
        """Get all observed cells."""
        return list(self._cells.values())

    def get_obstacles(self) -> list[GridCell]:
        """Get all obstacle cells."""
        return [c for c in self._cells.values() if c.state == CellState.OCCUPIED]

    def get_bounds(self) -> tuple[int, int, int, int]:
        """Get the bounding box of observed cells (min_x, min_y, max_x, max_y)."""
        if not self._cells:
            return 0, 0, 0, 0

        xs: list[int] = [k[0] for k in self._cells.keys()]
        ys: list[int] = [k[1] for k in self._cells.keys()]
        return min(xs), min(ys), max(xs), max(ys)

    def stats(self) -> dict[str, Any]:
        """Get grid statistics."""
        min_x, min_y, max_x, max_y = self.get_bounds()
        return {
            "total_cells": self.cell_count,
            "free_cells": self.free_count,
            "obstacle_cells": self.obstacle_count,
            "unknown_cells": self.cell_count - self.free_count - self.obstacle_count,
            "cell_size_m": self._cell_size_m,
            "bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
            "area_sq_m": self.cell_count * self._cell_size_m ** 2
        }


class PathPlanner:
    """
    A* pathfinder on the occupancy grid. Finds shortest traversable
    path between two GPS positions, avoiding known obstacles.

    Features:
        - 8-directional movement (cardinal + diagonal)
        - Unknown cells are traversable but penalized
        - Obstacle buffer zone prevents paths that graze obstacles
        - Path simplification removes redundant intermediate waypoints
        - Dynamic replanning when new obstacles are detected mid-route
    """

    DIRECTIONS: list[tuple[int, int, float]] = [
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (1, 1, 1.414),
        (1, -1, 1.414),
        (-1, 1, 1.414),
        (-1, -1, 1.414),
    ]

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        cell_size: float = config.get("path_planner", "cell_size_m", default=0.5)
        home_lat: float = config.get("navigation", "home_lat", default=0.0)
        home_lon: float = config.get("navigation", "home_lon", default=0.0)
        self._unknown_penalty: float = config.get("path_planner", "unknown_penalty", default=1.5)
        self._max_iterations: int = config.get("path_planner", "max_iterations", default=10000)
        self._simplify_paths: bool = config.get("path_planner", "simplify_paths", default=True)
        self._simplify_tolerance_m: float = config.get("path_planner", "simplify_tolerance_m", default=1.0)

        self._grid: OccupancyGrid = OccupancyGrid(
            cell_size_m=cell_size,
            home_lat=home_lat,
            home_lon=home_lon
        )

        self._db: Optional[Any] = None
        self._total_plans: int = 0
        self._successful_plans: int = 0
        self._failed_plans: int = 0

        logger.info(
            "Path planner initialized — cell=%.1fm, unknown_penalty=%.1f, max_iter=%d",
            cell_size, self._unknown_penalty, self._max_iterations
        )

    def bind_db(self, db: Any) -> None:
        """Bind database for persistent grid storage."""
        self._db = db
        self._load_grid_from_db()

    @property
    def grid(self) -> OccupancyGrid:
        return self._grid

    def plan_path(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float
    ) -> PlannedPath:
        """
        Plan a path from start to goal GPS coordinates.
        Returns a PlannedPath with GPS waypoints.
        """
        start_time: float = time.time()
        self._total_plans += 1

        start_x, start_y = self._grid.gps_to_grid(start_lat, start_lon)
        goal_x, goal_y = self._grid.gps_to_grid(goal_lat, goal_lon)

        if start_x == goal_x and start_y == goal_y:
            return PlannedPath(
                waypoints=[(goal_lat, goal_lon)],
                grid_path=[(goal_x, goal_y)],
                distance_m=0.0,
                cell_count=1,
                planning_time_ms=0.0,
                success=True,
                message="Start and goal are in the same cell"
            )

        if not self._grid.is_traversable(goal_x, goal_y):
            goal_x, goal_y = self._find_nearest_free(goal_x, goal_y, radius=5)
            if goal_x is None:
                self._failed_plans += 1
                return PlannedPath(
                    planning_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    message="Goal position is blocked and no nearby free cell found"
                )

        grid_path: Optional[list[tuple[int, int]]] = self._astar(
            start_x, start_y, goal_x, goal_y
        )

        if grid_path is None:
            self._failed_plans += 1
            elapsed: float = (time.time() - start_time) * 1000
            return PlannedPath(
                planning_time_ms=elapsed,
                success=False,
                message=f"No path found ({self._max_iterations} iterations exhausted)"
            )

        if self._simplify_paths and len(grid_path) > 2:
            grid_path = self._simplify_grid_path(grid_path)

        gps_waypoints: list[tuple[float, float]] = [
            self._grid.grid_to_gps(x, y) for x, y in grid_path
        ]

        distance: float = self._calculate_path_distance(grid_path)
        elapsed = (time.time() - start_time) * 1000

        self._successful_plans += 1

        logger.info(
            "Path planned: %d waypoints, %.1fm, %.1fms (%d->%d, %d->%d)",
            len(gps_waypoints), distance, elapsed,
            start_x, goal_x, start_y, goal_y
        )

        return PlannedPath(
            waypoints=gps_waypoints,
            grid_path=grid_path,
            distance_m=distance,
            cell_count=len(grid_path),
            planning_time_ms=elapsed,
            success=True,
            message=f"Path found: {len(gps_waypoints)} waypoints, {distance:.1f}m"
        )

    def _astar(
        self,
        start_x: int,
        start_y: int,
        goal_x: int,
        goal_y: int
    ) -> Optional[list[tuple[int, int]]]:
        """A* pathfinding. Returns grid path or None if no path exists."""
        start_node: PathNode = PathNode(
            x=start_x, y=start_y, g_cost=0.0,
            h_cost=self._heuristic(start_x, start_y, goal_x, goal_y)
        )

        open_set: list[PathNode] = [start_node]
        closed_set: set[tuple[int, int]] = set()
        g_scores: dict[tuple[int, int], float] = {(start_x, start_y): 0.0}
        iterations: int = 0

        while open_set and iterations < self._max_iterations:
            iterations += 1
            current: PathNode = heapq.heappop(open_set)

            if current.x == goal_x and current.y == goal_y:
                return self._reconstruct_path(current)

            current_key: tuple[int, int] = (current.x, current.y)
            if current_key in closed_set:
                continue
            closed_set.add(current_key)

            for dx, dy, base_cost in self.DIRECTIONS:
                nx: int = current.x + dx
                ny: int = current.y + dy
                neighbor_key: tuple[int, int] = (nx, ny)

                if neighbor_key in closed_set:
                    continue

                if not self._grid.is_traversable(nx, ny):
                    continue

                move_cost: float = base_cost * self._cell_size_cost(nx, ny)
                new_g: float = current.g_cost + move_cost

                if new_g < g_scores.get(neighbor_key, float("inf")):
                    g_scores[neighbor_key] = new_g
                    neighbor: PathNode = PathNode(
                        x=nx, y=ny,
                        g_cost=new_g,
                        h_cost=self._heuristic(nx, ny, goal_x, goal_y),
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor)

        logger.debug("A* exhausted after %d iterations", iterations)
        return None

    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Octile distance heuristic (consistent for 8-directional movement)."""
        dx: int = abs(x2 - x1)
        dy: int = abs(y2 - y1)
        return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

    def _cell_size_cost(self, x: int, y: int) -> float:
        """Movement cost multiplier for a cell. Unknown cells get a penalty."""
        cell: GridCell = self._grid.get_cell(x, y)
        if cell.state == CellState.UNKNOWN:
            return self._unknown_penalty
        return 1.0

    def _reconstruct_path(self, node: PathNode) -> list[tuple[int, int]]:
        """Walk back through parent nodes to build the path."""
        path: list[tuple[int, int]] = []
        current: Optional[PathNode] = node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()
        return path

    def _simplify_grid_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Simplify a grid path by removing intermediate points that lie
        on a straight line. Uses the Ramer-Douglas-Peucker algorithm
        adapted for grid coordinates.
        """
        if len(path) <= 2:
            return path

        tolerance: float = self._simplify_tolerance_m / self._grid.cell_size_m

        return self._rdp_simplify(path, tolerance)

    def _rdp_simplify(
        self,
        points: list[tuple[int, int]],
        tolerance: float
    ) -> list[tuple[int, int]]:
        """Ramer-Douglas-Peucker line simplification."""
        if len(points) <= 2:
            return points

        max_dist: float = 0.0
        max_idx: int = 0

        start: tuple[int, int] = points[0]
        end: tuple[int, int] = points[-1]

        for i in range(1, len(points) - 1):
            dist: float = self._point_to_line_distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > tolerance:
            left: list[tuple[int, int]] = self._rdp_simplify(points[:max_idx + 1], tolerance)
            right: list[tuple[int, int]] = self._rdp_simplify(points[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [start, end]

    @staticmethod
    def _point_to_line_distance(
        point: tuple[int, int],
        line_start: tuple[int, int],
        line_end: tuple[int, int]
    ) -> float:
        """Perpendicular distance from a point to a line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx: float = float(x2 - x1)
        dy: float = float(y2 - y1)

        length_sq: float = dx * dx + dy * dy
        if length_sq == 0.0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t: float = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x: float = x1 + t * dx
        proj_y: float = y1 + t * dy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _find_nearest_free(
        self,
        x: int,
        y: int,
        radius: int = 5
    ) -> tuple[Optional[int], Optional[int]]:
        """Find the nearest traversable cell to a blocked position."""
        best_dist: float = float("inf")
        best_x: Optional[int] = None
        best_y: Optional[int] = None

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx: int = x + dx
                ny: int = y + dy
                if self._grid.is_traversable(nx, ny):
                    dist: float = math.sqrt(dx * dx + dy * dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_x = nx
                        best_y = ny

        return best_x, best_y

    def _calculate_path_distance(self, grid_path: list[tuple[int, int]]) -> float:
        """Calculate total path distance in meters."""
        if len(grid_path) < 2:
            return 0.0

        total: float = 0.0
        for i in range(len(grid_path) - 1):
            dx: float = float(grid_path[i + 1][0] - grid_path[i][0])
            dy: float = float(grid_path[i + 1][1] - grid_path[i][1])
            total += math.sqrt(dx * dx + dy * dy) * self._grid.cell_size_m

        return total

    def update_from_position(self, lat: float, lon: float) -> None:
        """Mark the rover's current position as free."""
        self._grid.mark_free(lat, lon)

    def update_from_obstacle(
        self,
        rover_lat: float,
        rover_lon: float,
        distance_cm: float,
        heading_deg: float,
        sensor_offset_deg: float = 0.0
    ) -> None:
        """Record an obstacle detection from the ultrasonic sensors."""
        self._grid.mark_obstacle(
            rover_lat, rover_lon, distance_cm, heading_deg, sensor_offset_deg
        )
        self._save_obstacle_to_db(rover_lat, rover_lon, distance_cm, heading_deg, sensor_offset_deg)

    def _save_obstacle_to_db(
        self,
        rover_lat: float,
        rover_lon: float,
        distance_cm: float,
        heading_deg: float,
        sensor_offset_deg: float
    ) -> None:
        """Persist obstacle to database."""
        if self._db is None:
            return

        total_heading: float = (heading_deg + sensor_offset_deg) % 360.0
        heading_rad: float = math.radians(total_heading)
        distance_m: float = distance_cm / 100.0

        obs_meters_east: float = distance_m * math.sin(heading_rad)
        obs_meters_north: float = distance_m * math.cos(heading_rad)

        delta_lat: float = obs_meters_north / (EARTH_RADIUS_M * math.pi / 180.0)
        delta_lon: float = obs_meters_east / (
            EARTH_RADIUS_M * math.pi / 180.0 * math.cos(math.radians(rover_lat))
        )

        obs_lat: float = rover_lat + delta_lat
        obs_lon: float = rover_lon + delta_lon
        gx, gy = self._grid.gps_to_grid(obs_lat, obs_lon)

        try:
            self._db.update_grid_cell(
                grid_x=gx, grid_y=gy, state="occupied",
                confidence=1.0, gps_lat=obs_lat, gps_lon=obs_lon
            )
        except Exception as e:
            logger.error("Failed to save obstacle to DB: %s", e)

    def _load_grid_from_db(self) -> None:
        """Load persisted grid cells from database."""
        if self._db is None:
            return

        try:
            rows: list[dict[str, Any]] = self._db.get_occupancy_grid()
            loaded: int = 0

            for row in rows:
                state_str: str = row.get("state", "unknown")
                try:
                    state: CellState = CellState(state_str)
                except ValueError:
                    state = CellState.UNKNOWN

                self._grid.set_cell(
                    x=row["grid_x"],
                    y=row["grid_y"],
                    state=state,
                    confidence=row.get("confidence", 0.0),
                    gps_lat=row.get("gps_lat", 0.0),
                    gps_lon=row.get("gps_lon", 0.0)
                )
                loaded += 1

            logger.info("Loaded %d grid cells from database (%d obstacles)",
                        loaded, self._grid.obstacle_count)

        except Exception as e:
            logger.error("Failed to load grid from database: %s", e)

    def save_grid_to_db(self) -> int:
        """Persist current grid to database. Called during shutdown or maintenance."""
        if self._db is None:
            return 0

        saved: int = 0
        for cell in self._grid.get_all_cells():
            try:
                self._db.update_grid_cell(
                    grid_x=cell.x, grid_y=cell.y,
                    state=cell.state.value,
                    confidence=cell.confidence,
                    gps_lat=cell.gps_lat, gps_lon=cell.gps_lon
                )
                saved += 1
            except Exception as e:
                logger.error("Failed to save grid cell (%d,%d): %s", cell.x, cell.y, e)

        logger.info("Saved %d grid cells to database", saved)
        return saved

    def clear_stale(self, max_age_days: int = 7) -> int:
        """Clear obstacles older than max_age_days."""
        cleared: int = self._grid.clear_stale_obstacles(max_age_days * 86400.0)

        if self._db is not None and cleared > 0:
            try:
                self._db.clear_stale_obstacles(max_age_days)
            except Exception as e:
                logger.error("Failed to clear stale obstacles in DB: %s", e)

        return cleared

    @property
    def total_plans(self) -> int:
        return self._total_plans

    @property
    def success_rate(self) -> float:
        if self._total_plans == 0:
            return 0.0
        return self._successful_plans / self._total_plans

    def stats(self) -> dict[str, Any]:
        grid_stats: dict[str, Any] = self._grid.stats()
        grid_stats["total_plans"] = self._total_plans
        grid_stats["successful_plans"] = self._successful_plans
        grid_stats["failed_plans"] = self._failed_plans
        grid_stats["success_rate"] = round(self.success_rate, 2)
        return grid_stats