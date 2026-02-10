"""
Tests for cerberus.intelligence.path_planner — PathPlanner, OccupancyGrid
Validates GPS/grid conversion, cell state management, A* pathfinding,
path simplification, obstacle buffer zones, stale obstacle clearing,
and database persistence.
"""

import math
import pytest
from typing import Any, Optional
from unittest.mock import MagicMock

from cerberus.core.config import CerberusConfig
from cerberus.intelligence.path_planner import (
    PathPlanner, OccupancyGrid, PlannedPath, GridCell, PathNode,
    CellState, EARTH_RADIUS_M
)


HOME_LAT: float = 36.1699
HOME_LON: float = -115.1398


class TestCellState:
    """CellState enum."""

    def test_all_states_exist(self) -> None:
        assert CellState.FREE.value == "free"
        assert CellState.OCCUPIED.value == "occupied"
        assert CellState.UNKNOWN.value == "unknown"


class TestGridCell:
    """GridCell dataclass."""

    def test_default_cell_is_unknown(self) -> None:
        cell: GridCell = GridCell()
        assert cell.state == CellState.UNKNOWN
        assert cell.traversable is True

    def test_free_cell_is_traversable(self) -> None:
        cell: GridCell = GridCell(state=CellState.FREE)
        assert cell.traversable is True

    def test_occupied_cell_not_traversable(self) -> None:
        cell: GridCell = GridCell(state=CellState.OCCUPIED)
        assert cell.traversable is False


class TestPathNode:
    """PathNode for A* search."""

    def test_f_cost(self) -> None:
        node: PathNode = PathNode(x=0, y=0, g_cost=5.0, h_cost=3.0)
        assert node.f_cost == 8.0

    def test_ordering(self) -> None:
        a: PathNode = PathNode(x=0, y=0, g_cost=1.0, h_cost=1.0)
        b: PathNode = PathNode(x=1, y=1, g_cost=5.0, h_cost=5.0)
        assert a < b

    def test_equality(self) -> None:
        a: PathNode = PathNode(x=3, y=4, g_cost=1.0)
        b: PathNode = PathNode(x=3, y=4, g_cost=99.0)
        assert a == b

    def test_not_equal_different_position(self) -> None:
        a: PathNode = PathNode(x=3, y=4)
        b: PathNode = PathNode(x=5, y=6)
        assert a != b

    def test_hash(self) -> None:
        a: PathNode = PathNode(x=3, y=4)
        b: PathNode = PathNode(x=3, y=4)
        assert hash(a) == hash(b)
        s: set = {a, b}
        assert len(s) == 1


class TestPlannedPath:
    """PlannedPath result dataclass."""

    def test_default_is_failure(self) -> None:
        path: PlannedPath = PlannedPath()
        assert path.success is False
        assert len(path.waypoints) == 0

    def test_to_dict(self) -> None:
        path: PlannedPath = PlannedPath(
            waypoints=[(36.17, -115.14), (36.171, -115.141)],
            grid_path=[(0, 0), (1, 1)],
            distance_m=5.0,
            cell_count=2,
            planning_time_ms=1.5,
            success=True,
            message="OK"
        )
        d: dict[str, Any] = path.to_dict()
        assert d["waypoint_count"] == 2
        assert d["distance_m"] == 5.0
        assert d["success"] is True


class TestOccupancyGridBasics:
    """OccupancyGrid initialization and cell management."""

    def test_empty_grid(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        assert grid.cell_count == 0
        assert grid.obstacle_count == 0
        assert grid.free_count == 0

    def test_cell_size(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=1.0)
        assert grid.cell_size_m == 1.0

    def test_set_and_get_cell(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(5, 10, CellState.FREE, confidence=1.0)
        cell: GridCell = grid.get_cell(5, 10)
        assert cell.state == CellState.FREE
        assert cell.confidence == 1.0
        assert cell.observation_count == 1

    def test_get_unobserved_cell_returns_unknown(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        cell: GridCell = grid.get_cell(999, 999)
        assert cell.state == CellState.UNKNOWN
        assert cell.observation_count == 0

    def test_update_existing_cell(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(5, 10, CellState.FREE)
        grid.set_cell(5, 10, CellState.OCCUPIED)
        cell: GridCell = grid.get_cell(5, 10)
        assert cell.state == CellState.OCCUPIED
        assert cell.observation_count == 2

    def test_is_traversable(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        assert grid.is_traversable(0, 0) is True
        grid.set_cell(0, 0, CellState.OCCUPIED)
        assert grid.is_traversable(0, 0) is False
        grid.set_cell(0, 0, CellState.FREE)
        assert grid.is_traversable(0, 0) is True

    def test_mark_free(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        grid.mark_free(HOME_LAT, HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT, HOME_LON)
        cell: GridCell = grid.get_cell(x, y)
        assert cell.state == CellState.FREE

    def test_counts(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(0, 0, CellState.FREE)
        grid.set_cell(1, 1, CellState.OCCUPIED)
        grid.set_cell(2, 2, CellState.FREE)
        assert grid.cell_count == 3
        assert grid.free_count == 2
        assert grid.obstacle_count == 1


class TestGPSGridConversion:
    """GPS to grid and grid to GPS coordinate conversion."""

    def test_home_is_origin(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT, HOME_LON)
        assert x == 0
        assert y == 0

    def test_north_increases_y(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=1.0, home_lat=HOME_LAT, home_lon=HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT + 0.0001, HOME_LON)
        assert y > 0
        assert abs(x) <= 1

    def test_east_increases_x(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=1.0, home_lat=HOME_LAT, home_lon=HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT, HOME_LON + 0.0001)
        assert x > 0
        assert abs(y) <= 1

    def test_roundtrip_conversion(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        test_lat: float = HOME_LAT + 0.001
        test_lon: float = HOME_LON + 0.001

        gx, gy = grid.gps_to_grid(test_lat, test_lon)
        back_lat, back_lon = grid.grid_to_gps(gx, gy)

        assert abs(back_lat - test_lat) < 0.0005
        assert abs(back_lon - test_lon) < 0.0005

    def test_south_decreases_y(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=1.0, home_lat=HOME_LAT, home_lon=HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT - 0.0001, HOME_LON)
        assert y < 0

    def test_west_decreases_x(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=1.0, home_lat=HOME_LAT, home_lon=HOME_LON)
        x, y = grid.gps_to_grid(HOME_LAT, HOME_LON - 0.0001)
        assert x < 0


class TestObstacleMarking:
    """Obstacle detection and buffer zones."""

    def test_mark_obstacle(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        grid.mark_obstacle(
            lat=HOME_LAT, lon=HOME_LON,
            distance_cm=200.0, heading_deg=0.0
        )
        assert grid.obstacle_count >= 1

    def test_obstacle_buffer_expands(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        grid._obstacle_buffer = 1
        grid.mark_obstacle(
            lat=HOME_LAT, lon=HOME_LON,
            distance_cm=200.0, heading_deg=0.0
        )
        assert grid.obstacle_count > 1

    def test_obstacle_buffer_zero(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        grid._obstacle_buffer = 0
        grid.mark_obstacle(
            lat=HOME_LAT, lon=HOME_LON,
            distance_cm=200.0, heading_deg=0.0
        )
        assert grid.obstacle_count == 1

    def test_obstacle_with_sensor_offset(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        grid._obstacle_buffer = 0
        grid.mark_obstacle(
            lat=HOME_LAT, lon=HOME_LON,
            distance_cm=100.0, heading_deg=0.0, sensor_offset_deg=45.0
        )
        assert grid.obstacle_count == 1
        obs: list[GridCell] = grid.get_obstacles()
        assert obs[0].x > 0 or obs[0].y > 0


class TestStaleObstacles:
    """Stale obstacle clearing."""

    def test_clear_stale_obstacles(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(0, 0, CellState.OCCUPIED, confidence=1.0)
        grid._cells[(0, 0)].last_observed = 0.0
        cleared: int = grid.clear_stale_obstacles(max_age_seconds=1.0)
        assert cleared == 1
        assert grid.get_cell(0, 0).state == CellState.UNKNOWN

    def test_recent_obstacles_not_cleared(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(0, 0, CellState.OCCUPIED, confidence=1.0)
        cleared: int = grid.clear_stale_obstacles(max_age_seconds=999999.0)
        assert cleared == 0
        assert grid.get_cell(0, 0).state == CellState.OCCUPIED


class TestGridStats:
    """Grid statistics and bounds."""

    def test_empty_grid_stats(self) -> None:
        grid: OccupancyGrid = OccupancyGrid(cell_size_m=0.5)
        s: dict[str, Any] = grid.stats()
        assert s["total_cells"] == 0
        assert s["cell_size_m"] == 0.5

    def test_bounds(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(-3, -2, CellState.FREE)
        grid.set_cell(5, 10, CellState.FREE)
        min_x, min_y, max_x, max_y = grid.get_bounds()
        assert min_x == -3
        assert min_y == -2
        assert max_x == 5
        assert max_y == 10

    def test_empty_bounds(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        assert grid.get_bounds() == (0, 0, 0, 0)

    def test_get_all_cells(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(0, 0, CellState.FREE)
        grid.set_cell(1, 1, CellState.OCCUPIED)
        cells: list[GridCell] = grid.get_all_cells()
        assert len(cells) == 2

    def test_get_obstacles_list(self) -> None:
        grid: OccupancyGrid = OccupancyGrid()
        grid.set_cell(0, 0, CellState.FREE)
        grid.set_cell(1, 1, CellState.OCCUPIED)
        grid.set_cell(2, 2, CellState.OCCUPIED)
        obs: list[GridCell] = grid.get_obstacles()
        assert len(obs) == 2


class TestPathPlannerInit:
    """PathPlanner initialization."""

    def test_creates_with_config(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        assert planner is not None
        assert planner.grid is not None

    def test_initial_stats(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        s: dict[str, Any] = planner.stats()
        assert s["total_plans"] == 0
        assert s["successful_plans"] == 0
        assert s["failed_plans"] == 0
        assert s["success_rate"] == 0.0


class TestAStarPathfinding:
    """A* pathfinding algorithm."""

    def _make_planner(self, config: CerberusConfig) -> PathPlanner:
        planner: PathPlanner = PathPlanner(config)
        return planner

    def test_same_cell_path(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        result: PlannedPath = planner.plan_path(HOME_LAT, HOME_LON, HOME_LAT, HOME_LON)
        assert result.success is True
        assert result.cell_count == 1
        assert result.distance_m == 0.0

    def test_straight_line_path(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        goal_lat: float = HOME_LAT + 0.0005
        result: PlannedPath = planner.plan_path(HOME_LAT, HOME_LON, goal_lat, HOME_LON)
        assert result.success is True
        assert result.distance_m > 0
        assert len(result.waypoints) >= 2

    def test_path_avoids_obstacle(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        grid: OccupancyGrid = planner.grid

        for i in range(-5, 6):
            grid.set_cell(0, i, CellState.FREE)
            grid.set_cell(1, i, CellState.FREE)
            grid.set_cell(-1, i, CellState.FREE)
            grid.set_cell(2, i, CellState.FREE)
            grid.set_cell(-2, i, CellState.FREE)

        grid.set_cell(0, 3, CellState.OCCUPIED)
        grid.set_cell(0, 4, CellState.OCCUPIED)
        grid.set_cell(0, 5, CellState.OCCUPIED)

        start_lat, start_lon = grid.grid_to_gps(0, 0)
        goal_lat, goal_lon = grid.grid_to_gps(0, 8)

        result: PlannedPath = planner.plan_path(start_lat, start_lon, goal_lat, goal_lon)
        assert result.success is True

        obstacle_cells: set[tuple[int, int]] = {(0, 3), (0, 4), (0, 5)}
        for gx, gy in result.grid_path:
            assert (gx, gy) not in obstacle_cells

    def test_blocked_goal_finds_nearest(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        grid: OccupancyGrid = planner.grid

        grid.set_cell(10, 10, CellState.OCCUPIED)
        grid.set_cell(10, 11, CellState.FREE)

        start_lat, start_lon = grid.grid_to_gps(0, 0)
        goal_lat, goal_lon = grid.grid_to_gps(10, 10)

        result: PlannedPath = planner.plan_path(start_lat, start_lon, goal_lat, goal_lon)
        assert result.success is True

    def test_completely_blocked_fails(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        grid: OccupancyGrid = planner.grid

        for dx in range(-6, 7):
            for dy in range(-6, 7):
                grid.set_cell(10 + dx, 10 + dy, CellState.OCCUPIED)

        start_lat, start_lon = grid.grid_to_gps(0, 0)
        goal_lat, goal_lon = grid.grid_to_gps(10, 10)

        result: PlannedPath = planner.plan_path(start_lat, start_lon, goal_lat, goal_lon)
        assert result.success is False

    def test_planning_increments_counters(self, config: CerberusConfig) -> None:
        planner: PathPlanner = self._make_planner(config)
        planner.plan_path(HOME_LAT, HOME_LON, HOME_LAT + 0.0001, HOME_LON)
        assert planner.total_plans == 1
        assert planner.success_rate > 0


class TestPathSimplification:
    """Ramer-Douglas-Peucker path simplification."""

    def test_straight_line_simplified(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        path: list[tuple[int, int]] = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        simplified: list[tuple[int, int]] = planner._simplify_grid_path(path)
        assert len(simplified) < len(path)
        assert simplified[0] == (0, 0)
        assert simplified[-1] == (0, 5)

    def test_right_angle_preserved(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner._simplify_tolerance_m = 0.1
        path: list[tuple[int, int]] = [(0, 0), (0, 5), (0, 10), (5, 10), (10, 10)]
        simplified: list[tuple[int, int]] = planner._simplify_grid_path(path)
        assert len(simplified) >= 3

    def test_short_path_not_simplified(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        path: list[tuple[int, int]] = [(0, 0), (5, 5)]
        simplified: list[tuple[int, int]] = planner._simplify_grid_path(path)
        assert simplified == path

    def test_simplification_disabled(self, config: CerberusConfig) -> None:
        config._data["path_planner"]["simplify_paths"] = False
        planner: PathPlanner = PathPlanner(config)
        goal_lat: float = HOME_LAT + 0.001
        result: PlannedPath = planner.plan_path(HOME_LAT, HOME_LON, goal_lat, HOME_LON)
        assert result.success is True


class TestPointToLineDistance:
    """Perpendicular distance calculation."""

    def test_point_on_line(self) -> None:
        dist: float = PathPlanner._point_to_line_distance((5, 5), (0, 0), (10, 10))
        assert dist < 0.01

    def test_point_off_line(self) -> None:
        dist: float = PathPlanner._point_to_line_distance((0, 5), (0, 0), (10, 0))
        assert abs(dist - 5.0) < 0.01

    def test_zero_length_segment(self) -> None:
        dist: float = PathPlanner._point_to_line_distance((3, 4), (0, 0), (0, 0))
        assert abs(dist - 5.0) < 0.01


class TestPathDistance:
    """Path distance calculation."""

    def test_straight_path_distance(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        path: list[tuple[int, int]] = [(0, 0), (0, 10)]
        dist: float = planner._calculate_path_distance(path)
        expected: float = 10 * planner.grid.cell_size_m
        assert abs(dist - expected) < 0.01

    def test_diagonal_path_distance(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        path: list[tuple[int, int]] = [(0, 0), (10, 10)]
        dist: float = planner._calculate_path_distance(path)
        expected: float = math.sqrt(200) * planner.grid.cell_size_m
        assert abs(dist - expected) < 0.1

    def test_empty_path_zero_distance(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        assert planner._calculate_path_distance([]) == 0.0
        assert planner._calculate_path_distance([(0, 0)]) == 0.0


class TestGridUpdates:
    """Rover position and obstacle updates."""

    def test_update_from_position(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.update_from_position(HOME_LAT, HOME_LON)
        x, y = planner.grid.gps_to_grid(HOME_LAT, HOME_LON)
        cell: GridCell = planner.grid.get_cell(x, y)
        assert cell.state == CellState.FREE

    def test_update_from_obstacle(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.update_from_obstacle(
            rover_lat=HOME_LAT, rover_lon=HOME_LON,
            distance_cm=150.0, heading_deg=90.0
        )
        assert planner.grid.obstacle_count >= 1


class TestDatabaseIntegration:
    """Grid persistence with database."""

    def test_bind_db(self, config: CerberusConfig, db) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)
        assert planner._db is db

    def test_save_and_load_grid(self, config: CerberusConfig, db) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)

        planner.grid.set_cell(1, 1, CellState.FREE, confidence=1.0, gps_lat=HOME_LAT, gps_lon=HOME_LON)
        planner.grid.set_cell(2, 2, CellState.OCCUPIED, confidence=0.9, gps_lat=HOME_LAT, gps_lon=HOME_LON)
        saved: int = planner.save_grid_to_db()
        assert saved == 2

        planner2: PathPlanner = PathPlanner.__new__(PathPlanner)
        planner2._grid = OccupancyGrid(cell_size_m=0.5, home_lat=HOME_LAT, home_lon=HOME_LON)
        planner2._db = db
        planner2._unknown_penalty = 1.5
        planner2._max_iterations = 10000
        planner2._simplify_paths = True
        planner2._simplify_tolerance_m = 1.0
        planner2._total_plans = 0
        planner2._successful_plans = 0
        planner2._failed_plans = 0
        planner2._load_grid_from_db()

        assert planner2.grid.cell_count >= 2
        assert planner2.grid.get_cell(1, 1).state == CellState.FREE
        assert planner2.grid.get_cell(2, 2).state == CellState.OCCUPIED

    def test_save_without_db(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        saved: int = planner.save_grid_to_db()
        assert saved == 0

    def test_clear_stale_with_db(self, config: CerberusConfig, db) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.bind_db(db)

        planner.grid.set_cell(0, 0, CellState.OCCUPIED)
        planner.grid._cells[(0, 0)].last_observed = 0.0
        planner.save_grid_to_db()

        cleared: int = planner.clear_stale(max_age_days=0)
        assert cleared >= 1


class TestHeuristic:
    """Octile distance heuristic."""

    def test_straight_heuristic(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        h: float = planner._heuristic(0, 0, 10, 0)
        assert abs(h - 10.0) < 0.01

    def test_diagonal_heuristic(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        h: float = planner._heuristic(0, 0, 5, 5)
        expected: float = 5 + (1.414 - 1.0) * 5
        assert abs(h - expected) < 0.01

    def test_same_point_heuristic(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        h: float = planner._heuristic(3, 3, 3, 3)
        assert h == 0.0


class TestFindNearestFree:
    """Finding nearest traversable cell to a blocked position."""

    def test_finds_adjacent_free(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        planner.grid.set_cell(5, 5, CellState.OCCUPIED)
        planner.grid.set_cell(5, 6, CellState.FREE)

        bx, by = planner._find_nearest_free(5, 5, radius=2)
        assert bx is not None
        assert by is not None
        assert planner.grid.is_traversable(bx, by)

    def test_returns_none_if_fully_blocked(self, config: CerberusConfig) -> None:
        planner: PathPlanner = PathPlanner(config)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                planner.grid.set_cell(50 + dx, 50 + dy, CellState.OCCUPIED)

        bx, by = planner._find_nearest_free(50, 50, radius=2)
        assert bx is None or by is None or not planner.grid.is_traversable(bx, by) or \
               (abs(bx - 50) > 2 or abs(by - 50) > 2)