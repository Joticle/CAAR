"""
Cerberus Mission Planner
Loads, validates, executes, pauses, resumes, and aborts missions.
A mission is a YAML-defined sequence of tasks with waypoints, actions,
and conditions. The mission planner is the bridge between the brain
and the autonomous subsystems.
"""

import time
import logging
import threading
from typing import Any, Optional, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import yaml

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class MissionState(Enum):
    """Mission lifecycle states."""
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"


class TaskType(Enum):
    """Types of mission tasks."""
    NAVIGATE = "navigate"
    PATROL = "patrol"
    GRID_SCAN = "grid_scan"
    STATION_KEEP = "station_keep"
    SCAN = "scan"
    CAPTURE = "capture"
    WAIT = "wait"
    RTB = "rtb"


@dataclass
class MissionTask:
    """A single task within a mission."""
    task_id: int = 0
    task_type: TaskType = TaskType.WAIT
    name: str = ""
    waypoint_lat: float = 0.0
    waypoint_lon: float = 0.0
    waypoint_radius_m: float = 2.0
    speed: float = 0.5
    duration_seconds: float = 0.0
    heading_deg: Optional[float] = None
    action: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "waypoint_lat": round(self.waypoint_lat, 7),
            "waypoint_lon": round(self.waypoint_lon, 7),
            "speed": self.speed,
            "duration_seconds": self.duration_seconds,
            "action": self.action,
            "parameters": self.parameters,
            "completed": self.completed
        }


@dataclass
class MissionDefinition:
    """Complete mission definition loaded from YAML."""
    name: str = ""
    description: str = ""
    head: str = ""
    priority: int = 5
    repeat: bool = False
    max_repeats: int = 1
    tasks: list[MissionTask] = field(default_factory=list)
    home_lat: float = 0.0
    home_lon: float = 0.0
    max_duration_minutes: float = 60.0
    require_gps: bool = True
    require_camera: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "head": self.head,
            "priority": self.priority,
            "repeat": self.repeat,
            "max_repeats": self.max_repeats,
            "task_count": len(self.tasks),
            "max_duration_minutes": self.max_duration_minutes,
            "require_gps": self.require_gps,
            "require_camera": self.require_camera,
            "metadata": self.metadata
        }


@dataclass
class MissionStatus:
    """Current mission execution status."""
    state: MissionState = MissionState.IDLE
    mission_name: str = ""
    current_task_index: int = 0
    total_tasks: int = 0
    tasks_completed: int = 0
    elapsed_seconds: float = 0.0
    repeat_count: int = 0
    current_task_name: str = ""
    current_task_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "mission_name": self.mission_name,
            "current_task_index": self.current_task_index,
            "total_tasks": self.total_tasks,
            "tasks_completed": self.tasks_completed,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "repeat_count": self.repeat_count,
            "current_task_name": self.current_task_name,
            "current_task_type": self.current_task_type
        }


class MissionPlanner:
    """
    Manages mission lifecycle for Cerberus.
    Loads missions from YAML, validates prerequisites, and coordinates
    execution through the autonomous subsystems. Reports status to
    the brain and Dashboard via callbacks and MQTT.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._missions_dir: str = config.get("mission", "missions_dir", default="config/missions")
        self._max_duration: float = config.get("mission", "max_duration_minutes", default=60)
        self._home_lat: float = config.get("navigation", "home_lat", default=36.1699)
        self._home_lon: float = config.get("navigation", "home_lon", default=-115.1398)

        self._mission: Optional[MissionDefinition] = None
        self._state: MissionState = MissionState.IDLE
        self._current_task_index: int = 0
        self._repeat_count: int = 0
        self._start_time: float = 0.0

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()

        self._task_handlers: dict[TaskType, Callable[[MissionTask], bool]] = {}
        self._on_task_complete: Optional[Callable[[int, MissionTask], None]] = None
        self._on_mission_complete: Optional[Callable[[MissionDefinition], None]] = None
        self._on_mission_failed: Optional[Callable[[str], None]] = None
        self._on_status_update: Optional[Callable[[MissionStatus], None]] = None

        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None
        self._safety: Optional[Any] = None

        logger.info("Mission planner created — missions_dir=%s", self._missions_dir)

    def bind_db(self, db: Any) -> None:
        """Bind database for mission logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for status publishing."""
        self._mqtt = mqtt_client

    def bind_safety(self, safety: Any) -> None:
        """Bind safety watchdog for pre-mission checks."""
        self._safety = safety

    def register_task_handler(
        self,
        task_type: TaskType,
        handler: Callable[[MissionTask], bool]
    ) -> None:
        """
        Register a handler for a task type.
        Handler receives a MissionTask and returns True on success.
        The navigator registers for NAVIGATE.
        The patrol module registers for PATROL.
        The grid driver registers for GRID_SCAN.
        """
        self._task_handlers[task_type] = handler
        logger.info("Task handler registered: %s", task_type.value)

    def register_task_complete_callback(
        self,
        callback: Callable[[int, MissionTask], None]
    ) -> None:
        """Register callback for task completion: callback(index, task)."""
        self._on_task_complete = callback

    def register_mission_complete_callback(
        self,
        callback: Callable[[MissionDefinition], None]
    ) -> None:
        """Register callback for mission completion: callback(mission)."""
        self._on_mission_complete = callback

    def register_mission_failed_callback(
        self,
        callback: Callable[[str], None]
    ) -> None:
        """Register callback for mission failure: callback(reason)."""
        self._on_mission_failed = callback

    def register_status_callback(
        self,
        callback: Callable[[MissionStatus], None]
    ) -> None:
        """Register callback for status updates: callback(status)."""
        self._on_status_update = callback

    def load_mission_file(self, filename: str) -> bool:
        """
        Load a mission from a YAML file.
        filename: name relative to missions_dir or absolute path.
        """
        filepath: str = filename
        if not Path(filepath).is_absolute():
            filepath = str(Path(self._missions_dir) / filename)

        if not Path(filepath).exists():
            logger.error("Mission file not found: %s", filepath)
            return False

        try:
            with self._lock:
                self._state = MissionState.LOADING

            with open(filepath, "r", encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}

            mission: MissionDefinition = self._parse_mission(data)

            if not self._validate_mission(mission):
                with self._lock:
                    self._state = MissionState.IDLE
                return False

            with self._lock:
                self._mission = mission
                self._current_task_index = 0
                self._repeat_count = 0
                self._state = MissionState.READY

            logger.info(
                "Mission loaded: '%s' — %d tasks, head=%s",
                mission.name, len(mission.tasks), mission.head or "any"
            )
            return True

        except yaml.YAMLError as e:
            logger.error("Mission YAML parse error in %s: %s", filepath, e)
            with self._lock:
                self._state = MissionState.IDLE
            return False
        except Exception as e:
            logger.error("Failed to load mission %s: %s", filepath, e)
            with self._lock:
                self._state = MissionState.IDLE
            return False

    def load_mission(self, definition: MissionDefinition) -> bool:
        """Load a mission from a MissionDefinition object directly."""
        if not self._validate_mission(definition):
            return False

        with self._lock:
            self._mission = definition
            self._current_task_index = 0
            self._repeat_count = 0
            self._state = MissionState.READY

        logger.info(
            "Mission loaded: '%s' — %d tasks",
            definition.name, len(definition.tasks)
        )
        return True

    def _parse_mission(self, data: dict[str, Any]) -> MissionDefinition:
        """Parse a YAML dict into a MissionDefinition."""
        tasks: list[MissionTask] = []
        raw_tasks: list[dict] = data.get("tasks", [])

        for i, task_data in enumerate(raw_tasks):
            task_type_str: str = task_data.get("type", "wait")
            try:
                task_type: TaskType = TaskType(task_type_str)
            except ValueError:
                logger.warning("Unknown task type '%s' at index %d — defaulting to wait", task_type_str, i)
                task_type = TaskType.WAIT

            task: MissionTask = MissionTask(
                task_id=i,
                task_type=task_type,
                name=task_data.get("name", f"task_{i}"),
                waypoint_lat=task_data.get("lat", 0.0),
                waypoint_lon=task_data.get("lon", 0.0),
                waypoint_radius_m=task_data.get("radius_m", 2.0),
                speed=task_data.get("speed", 0.5),
                duration_seconds=task_data.get("duration", 0.0),
                heading_deg=task_data.get("heading", None),
                action=task_data.get("action", "none"),
                parameters=task_data.get("parameters", {})
            )
            tasks.append(task)

        return MissionDefinition(
            name=data.get("name", "unnamed_mission"),
            description=data.get("description", ""),
            head=data.get("head", ""),
            priority=data.get("priority", 5),
            repeat=data.get("repeat", False),
            max_repeats=data.get("max_repeats", 1),
            tasks=tasks,
            home_lat=data.get("home_lat", self._home_lat),
            home_lon=data.get("home_lon", self._home_lon),
            max_duration_minutes=data.get("max_duration_minutes", self._max_duration),
            require_gps=data.get("require_gps", True),
            require_camera=data.get("require_camera", False),
            metadata=data.get("metadata", {})
        )

    def _validate_mission(self, mission: MissionDefinition) -> bool:
        """Validate a mission before allowing execution."""
        if not mission.tasks:
            logger.error("Mission '%s' has no tasks", mission.name)
            return False

        if mission.max_duration_minutes <= 0:
            logger.error("Mission '%s' has invalid duration", mission.name)
            return False

        for task in mission.tasks:
            if task.task_type == TaskType.NAVIGATE:
                if task.waypoint_lat == 0.0 and task.waypoint_lon == 0.0:
                    logger.error(
                        "Mission '%s' task '%s' has no waypoint coordinates",
                        mission.name, task.name
                    )
                    return False

        logger.info("Mission '%s' validated — %d tasks", mission.name, len(mission.tasks))
        return True

    def start(self) -> bool:
        """Start executing the loaded mission."""
        with self._lock:
            if self._state != MissionState.READY:
                logger.warning(
                    "Cannot start mission — state is %s (must be READY)",
                    self._state.value
                )
                return False

            if self._mission is None:
                logger.error("No mission loaded")
                return False

        if self._safety is not None and not self._safety.is_safe_for_mission:
            logger.warning("Cannot start mission — safety violation active")
            return False

        self._running = True
        self._start_time = time.time()

        with self._lock:
            self._state = MissionState.RUNNING

        self._thread = threading.Thread(
            target=self._mission_loop,
            name="mission-executor",
            daemon=True
        )
        self._thread.start()

        logger.info("Mission '%s' STARTED", self._mission.name)
        self._log_event("mission_start", f"Mission '{self._mission.name}' started")
        self._publish_status()

        return True

    def pause(self) -> None:
        """Pause the current mission."""
        with self._lock:
            if self._state != MissionState.RUNNING:
                return
            self._state = MissionState.PAUSED

        self._pause_event.clear()
        logger.info("Mission PAUSED at task %d", self._current_task_index)
        self._publish_status()

    def resume(self) -> None:
        """Resume a paused mission."""
        with self._lock:
            if self._state != MissionState.PAUSED:
                return
            self._state = MissionState.RUNNING

        self._pause_event.set()
        logger.info("Mission RESUMED at task %d", self._current_task_index)
        self._publish_status()

    def abort(self, reason: str = "Abort requested") -> None:
        """Abort the current mission."""
        with self._lock:
            if self._state not in (MissionState.RUNNING, MissionState.PAUSED):
                return
            self._state = MissionState.ABORTING

        self._running = False
        self._pause_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

        with self._lock:
            self._state = MissionState.ABORTED

        logger.warning("Mission ABORTED: %s", reason)
        self._log_event("mission_abort", reason)
        self._publish_status()

    def _mission_loop(self) -> None:
        """Main mission execution loop."""
        logger.info("Mission execution loop started")

        while self._running:
            self._pause_event.wait()
            if not self._running:
                break

            if self._safety is not None and not self._safety.is_safe_for_mission:
                logger.warning("Safety violation during mission — aborting")
                self.abort(reason="Safety violation")
                return

            elapsed_min: float = (time.time() - self._start_time) / 60.0
            if self._mission and elapsed_min > self._mission.max_duration_minutes:
                logger.warning("Mission time limit exceeded (%.1f min)", elapsed_min)
                self.abort(reason="Time limit exceeded")
                return

            with self._lock:
                if self._mission is None:
                    break
                if self._current_task_index >= len(self._mission.tasks):
                    if self._mission.repeat and self._repeat_count < self._mission.max_repeats - 1:
                        self._repeat_count += 1
                        self._current_task_index = 0
                        for task in self._mission.tasks:
                            task.completed = False
                        logger.info(
                            "Mission repeat %d/%d",
                            self._repeat_count + 1, self._mission.max_repeats
                        )
                        continue
                    else:
                        break

                task: MissionTask = self._mission.tasks[self._current_task_index]

            success: bool = self._execute_task(task)

            with self._lock:
                task.completed = success
                task.completed_at = time.time()

                if self._on_task_complete:
                    try:
                        self._on_task_complete(self._current_task_index, task)
                    except Exception as e:
                        logger.error("Task complete callback error: %s", e)

                self._current_task_index += 1
                self._publish_status()

        self._complete_mission()

    def _execute_task(self, task: MissionTask) -> bool:
        """Execute a single mission task."""
        task.started_at = time.time()

        logger.info(
            "Executing task %d: %s (%s)",
            task.task_id, task.name, task.task_type.value
        )
        self._log_event(
            "task_start",
            f"Task {task.task_id}: {task.name} ({task.task_type.value})"
        )

        if task.task_type == TaskType.WAIT:
            return self._execute_wait(task)

        handler: Optional[Callable] = self._task_handlers.get(task.task_type)
        if handler is None:
            logger.warning(
                "No handler for task type '%s' — skipping task '%s'",
                task.task_type.value, task.name
            )
            return True

        try:
            return handler(task)
        except Exception as e:
            logger.error("Task '%s' execution error: %s", task.name, e)
            return False

    def _execute_wait(self, task: MissionTask) -> bool:
        """Execute a wait task — pause for specified duration."""
        duration: float = task.duration_seconds
        if duration <= 0:
            return True

        logger.info("Waiting %.1f seconds...", duration)
        end_time: float = time.time() + duration

        while time.time() < end_time and self._running:
            self._pause_event.wait(timeout=0.5)
            if not self._running:
                return False

        return True

    def _complete_mission(self) -> None:
        """Handle mission completion."""
        self._running = False

        with self._lock:
            was_aborting: bool = self._state == MissionState.ABORTING
            if not was_aborting:
                self._state = MissionState.COMPLETED

        if self._mission and not was_aborting:
            elapsed: float = time.time() - self._start_time
            tasks_done: int = sum(1 for t in self._mission.tasks if t.completed)

            logger.info(
                "Mission '%s' COMPLETED — %d/%d tasks in %.1fs",
                self._mission.name, tasks_done, len(self._mission.tasks), elapsed
            )
            self._log_event(
                "mission_complete",
                f"Mission '{self._mission.name}' completed: {tasks_done}/{len(self._mission.tasks)} tasks"
            )

            if self._on_mission_complete:
                try:
                    self._on_mission_complete(self._mission)
                except Exception as e:
                    logger.error("Mission complete callback error: %s", e)

        self._publish_status()

    def _log_event(self, event_type: str, message: str) -> None:
        """Log a mission event to the database."""
        if self._db is None:
            return
        try:
            self._db.log_mission_event(
                mission_name=self._mission.name if self._mission else "unknown",
                event_type=event_type,
                message=message
            )
        except Exception as e:
            logger.error("Failed to log mission event: %s", e)

    def _publish_status(self) -> None:
        """Publish current mission status to MQTT."""
        status: MissionStatus = self.status

        if self._on_status_update:
            try:
                self._on_status_update(status)
            except Exception as e:
                logger.error("Status update callback error: %s", e)

        if self._mqtt is not None:
            try:
                self._mqtt.publish_mission_status(status.to_dict())
            except Exception:
                pass

    @property
    def status(self) -> MissionStatus:
        """Get current mission status."""
        with self._lock:
            elapsed: float = 0.0
            if self._start_time > 0 and self._state == MissionState.RUNNING:
                elapsed = time.time() - self._start_time

            task_name: str = ""
            task_type: str = ""
            tasks_completed: int = 0

            if self._mission and self._current_task_index < len(self._mission.tasks):
                task: MissionTask = self._mission.tasks[self._current_task_index]
                task_name = task.name
                task_type = task.task_type.value
            if self._mission:
                tasks_completed = sum(1 for t in self._mission.tasks if t.completed)

            return MissionStatus(
                state=self._state,
                mission_name=self._mission.name if self._mission else "",
                current_task_index=self._current_task_index,
                total_tasks=len(self._mission.tasks) if self._mission else 0,
                tasks_completed=tasks_completed,
                elapsed_seconds=elapsed,
                repeat_count=self._repeat_count,
                current_task_name=task_name,
                current_task_type=task_type
            )

    @property
    def current_mission(self) -> Optional[MissionDefinition]:
        with self._lock:
            return self._mission

    @property
    def mission_state(self) -> MissionState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    def list_missions(self) -> list[str]:
        """List available mission files in the missions directory."""
        missions_path: Path = Path(self._missions_dir)
        if not missions_path.exists():
            return []
        return [f.name for f in missions_path.glob("*.yaml")]

    def stop(self) -> None:
        """Stop the mission planner. Called during shutdown."""
        if self._running:
            self.abort(reason="System shutdown")

    def __repr__(self) -> str:
        return (
            f"MissionPlanner(state={self._state.value}, "
            f"mission='{self._mission.name if self._mission else 'none'}', "
            f"task={self._current_task_index})"
        )