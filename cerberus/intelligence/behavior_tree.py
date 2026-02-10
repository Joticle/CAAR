"""
Cerberus Behavior Tree Engine
Priority-based reactive decision system that replaces linear mission
execution with intelligent, perception-driven behavior. The brain
evaluates the tree every cycle — higher-priority behaviors preempt
lower-priority ones. This is how Cerberus stops being a script
runner and starts being a thinker.

Priority Stack (highest first):
    1. Emergency Stop — safety violation, hardware fault, critical obstacle
    2. Return to Base — battery critical, thermal shutdown, lost GPS
    3. Threat Response — high-confidence threat during surveillance
    4. Obstacle Avoidance — obstacle in caution/critical zone during nav
    5. Active Investigation — interesting detection triggers closer look
    6. Mission Execution — follow the loaded mission plan
    7. Idle Patrol — no mission loaded, slow random patrol near home

Node Types:
    - Selector: runs children in order, succeeds on first child success
    - Sequence: runs children in order, fails on first child failure
    - Condition: evaluates a predicate, succeeds or fails
    - Action: executes a behavior, returns running/success/failure
    - Decorator: wraps a child node with modified behavior (cooldown, inverter, retry)
"""

import time
import logging
from typing import Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Result of evaluating a behavior tree node."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorPriority(Enum):
    """Named priority levels. Lower number = higher priority."""
    EMERGENCY_STOP = 1
    RETURN_TO_BASE = 2
    THREAT_RESPONSE = 3
    OBSTACLE_AVOIDANCE = 4
    ACTIVE_INVESTIGATION = 5
    MISSION_EXECUTION = 6
    IDLE_PATROL = 7


@dataclass
class BehaviorContext:
    """
    Shared state passed through the behavior tree each tick.
    Subsystems populate their fields, behaviors read them to
    make decisions. This is the rover's awareness snapshot.
    """
    # Time
    tick_count: int = 0
    timestamp: float = 0.0
    delta_time: float = 0.0

    # Safety
    battery_pct: float = 100.0
    battery_voltage: float = 12.6
    cpu_temp_c: float = 40.0
    safety_violation: bool = False
    safety_reason: str = ""

    # Navigation
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_fix: bool = False
    heading_deg: float = 0.0
    at_home: bool = False
    distance_to_home_m: float = 0.0

    # Obstacles
    obstacle_zone: str = "clear"
    obstacle_closest_cm: float = 400.0
    obstacle_path_clear: bool = True
    avoidance_direction: str = "none"
    avoidance_speed_limit: float = 1.0

    # Detections
    active_detection: bool = False
    detection_type: str = ""
    detection_label: str = ""
    detection_confidence: float = 0.0
    threat_detected: bool = False
    threat_level: str = "none"

    # Mission
    mission_active: bool = False
    mission_id: str = ""
    mission_paused: bool = False
    head_active: bool = False
    head_type: str = ""

    # Flags set by behaviors
    requesting_rtb: bool = False
    requesting_stop: bool = False
    requesting_investigation: bool = False
    investigation_lat: float = 0.0
    investigation_lon: float = 0.0

    # Arbitrary key-value store for inter-behavior communication
    blackboard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick_count,
            "battery_pct": self.battery_pct,
            "cpu_temp_c": self.cpu_temp_c,
            "gps_fix": self.gps_fix,
            "obstacle_zone": self.obstacle_zone,
            "obstacle_path_clear": self.obstacle_path_clear,
            "mission_active": self.mission_active,
            "active_detection": self.active_detection,
            "threat_detected": self.threat_detected,
            "requesting_rtb": self.requesting_rtb,
            "requesting_stop": self.requesting_stop
        }


# =================================================================
# BASE NODE
# =================================================================

class BehaviorNode(ABC):
    """Abstract base class for all behavior tree nodes."""

    def __init__(self, name: str = "") -> None:
        self._name: str = name or self.__class__.__name__
        self._status: NodeStatus = NodeStatus.FAILURE
        self._run_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> NodeStatus:
        return self._status

    @abstractmethod
    def tick(self, context: BehaviorContext) -> NodeStatus:
        """Evaluate this node. Must be implemented by subclasses."""
        ...

    def reset(self) -> None:
        """Reset node state. Override for stateful nodes."""
        self._status = NodeStatus.FAILURE

    def _record(self, status: NodeStatus) -> NodeStatus:
        """Record execution stats and return the status."""
        self._status = status
        self._run_count += 1
        if status == NodeStatus.SUCCESS:
            self._success_count += 1
        elif status == NodeStatus.FAILURE:
            self._failure_count += 1
        return status

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "type": self.__class__.__name__,
            "runs": self._run_count,
            "successes": self._success_count,
            "failures": self._failure_count,
            "last_status": self._status.value
        }


# =================================================================
# COMPOSITE NODES
# =================================================================

class Selector(BehaviorNode):
    """
    Tries children in order. Returns SUCCESS on the first child that
    succeeds. Returns RUNNING if a child is running. Returns FAILURE
    only if all children fail. This is the OR node.
    """

    def __init__(self, name: str = "", children: Optional[list[BehaviorNode]] = None) -> None:
        super().__init__(name)
        self._children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> "Selector":
        self._children.append(child)
        return self

    def tick(self, context: BehaviorContext) -> NodeStatus:
        for child in self._children:
            status: NodeStatus = child.tick(context)
            if status != NodeStatus.FAILURE:
                return self._record(status)
        return self._record(NodeStatus.FAILURE)

    def reset(self) -> None:
        super().reset()
        for child in self._children:
            child.reset()


class Sequence(BehaviorNode):
    """
    Runs children in order. Returns FAILURE on the first child that
    fails. Returns RUNNING if a child is running. Returns SUCCESS
    only if all children succeed. This is the AND node.
    """

    def __init__(self, name: str = "", children: Optional[list[BehaviorNode]] = None) -> None:
        super().__init__(name)
        self._children: list[BehaviorNode] = children or []

    def add_child(self, child: BehaviorNode) -> "Sequence":
        self._children.append(child)
        return self

    def tick(self, context: BehaviorContext) -> NodeStatus:
        for child in self._children:
            status: NodeStatus = child.tick(context)
            if status != NodeStatus.SUCCESS:
                return self._record(status)
        return self._record(NodeStatus.SUCCESS)

    def reset(self) -> None:
        super().reset()
        for child in self._children:
            child.reset()


class PrioritySelector(BehaviorNode):
    """
    Like Selector but always evaluates from the top on every tick.
    Higher-priority children can preempt a running lower-priority child.
    This is the core node for reactive behavior — safety always wins.
    """

    def __init__(self, name: str = "", children: Optional[list[BehaviorNode]] = None) -> None:
        super().__init__(name)
        self._children: list[BehaviorNode] = children or []
        self._running_child_idx: int = -1

    def add_child(self, child: BehaviorNode) -> "PrioritySelector":
        self._children.append(child)
        return self

    def tick(self, context: BehaviorContext) -> NodeStatus:
        for i, child in enumerate(self._children):
            status: NodeStatus = child.tick(context)

            if status == NodeStatus.RUNNING:
                if self._running_child_idx != -1 and self._running_child_idx != i:
                    self._children[self._running_child_idx].reset()
                    logger.info(
                        "Behavior preempted: %s interrupted by %s",
                        self._children[self._running_child_idx].name,
                        child.name
                    )
                self._running_child_idx = i
                return self._record(NodeStatus.RUNNING)

            if status == NodeStatus.SUCCESS:
                if self._running_child_idx != -1 and self._running_child_idx != i:
                    self._children[self._running_child_idx].reset()
                self._running_child_idx = -1
                return self._record(NodeStatus.SUCCESS)

        self._running_child_idx = -1
        return self._record(NodeStatus.FAILURE)

    def reset(self) -> None:
        super().reset()
        self._running_child_idx = -1
        for child in self._children:
            child.reset()


# =================================================================
# LEAF NODES
# =================================================================

class Condition(BehaviorNode):
    """
    Evaluates a predicate function against the context.
    Returns SUCCESS if true, FAILURE if false. Never RUNNING.
    """

    def __init__(self, name: str, predicate: Callable[[BehaviorContext], bool]) -> None:
        super().__init__(name)
        self._predicate: Callable[[BehaviorContext], bool] = predicate

    def tick(self, context: BehaviorContext) -> NodeStatus:
        try:
            result: bool = self._predicate(context)
            return self._record(NodeStatus.SUCCESS if result else NodeStatus.FAILURE)
        except Exception as e:
            logger.error("Condition '%s' raised exception: %s", self._name, e)
            return self._record(NodeStatus.FAILURE)


class Action(BehaviorNode):
    """
    Executes a behavior function. The function receives the context
    and returns a NodeStatus. Actions can be long-running (return RUNNING)
    or immediate (return SUCCESS/FAILURE).
    """

    def __init__(self, name: str, action: Callable[[BehaviorContext], NodeStatus]) -> None:
        super().__init__(name)
        self._action: Callable[[BehaviorContext], NodeStatus] = action

    def tick(self, context: BehaviorContext) -> NodeStatus:
        try:
            status: NodeStatus = self._action(context)
            return self._record(status)
        except Exception as e:
            logger.error("Action '%s' raised exception: %s", self._name, e)
            return self._record(NodeStatus.FAILURE)


# =================================================================
# DECORATOR NODES
# =================================================================

class Inverter(BehaviorNode):
    """Inverts the child result. SUCCESS becomes FAILURE and vice versa. RUNNING passes through."""

    def __init__(self, child: BehaviorNode, name: str = "") -> None:
        super().__init__(name or f"NOT({child.name})")
        self._child: BehaviorNode = child

    def tick(self, context: BehaviorContext) -> NodeStatus:
        status: NodeStatus = self._child.tick(context)
        if status == NodeStatus.SUCCESS:
            return self._record(NodeStatus.FAILURE)
        elif status == NodeStatus.FAILURE:
            return self._record(NodeStatus.SUCCESS)
        return self._record(NodeStatus.RUNNING)

    def reset(self) -> None:
        super().reset()
        self._child.reset()


class Cooldown(BehaviorNode):
    """
    Wraps a child node with a cooldown timer. After the child succeeds
    or runs, it won't be evaluated again until the cooldown expires.
    Prevents rapid re-triggering of expensive behaviors.
    """

    def __init__(self, child: BehaviorNode, cooldown_seconds: float, name: str = "") -> None:
        super().__init__(name or f"CD({child.name},{cooldown_seconds}s)")
        self._child: BehaviorNode = child
        self._cooldown: float = cooldown_seconds
        self._last_triggered: float = 0.0

    def tick(self, context: BehaviorContext) -> NodeStatus:
        now: float = time.time()
        elapsed: float = now - self._last_triggered

        if elapsed < self._cooldown:
            return self._record(NodeStatus.FAILURE)

        status: NodeStatus = self._child.tick(context)
        if status in (NodeStatus.SUCCESS, NodeStatus.RUNNING):
            self._last_triggered = now
        return self._record(status)

    def reset(self) -> None:
        super().reset()
        self._child.reset()


class Retry(BehaviorNode):
    """
    Retries a failing child up to max_attempts times before
    propagating the failure.
    """

    def __init__(self, child: BehaviorNode, max_attempts: int = 3, name: str = "") -> None:
        super().__init__(name or f"RETRY({child.name},{max_attempts})")
        self._child: BehaviorNode = child
        self._max_attempts: int = max_attempts
        self._attempts: int = 0

    def tick(self, context: BehaviorContext) -> NodeStatus:
        status: NodeStatus = self._child.tick(context)

        if status == NodeStatus.FAILURE:
            self._attempts += 1
            if self._attempts < self._max_attempts:
                self._child.reset()
                return self._record(NodeStatus.RUNNING)
            self._attempts = 0
            return self._record(NodeStatus.FAILURE)

        if status == NodeStatus.SUCCESS:
            self._attempts = 0

        return self._record(status)

    def reset(self) -> None:
        super().reset()
        self._attempts = 0
        self._child.reset()


class RepeatUntilFail(BehaviorNode):
    """Keeps ticking the child until it fails. Returns RUNNING while child succeeds."""

    def __init__(self, child: BehaviorNode, name: str = "") -> None:
        super().__init__(name or f"REPEAT({child.name})")
        self._child: BehaviorNode = child

    def tick(self, context: BehaviorContext) -> NodeStatus:
        status: NodeStatus = self._child.tick(context)

        if status == NodeStatus.FAILURE:
            return self._record(NodeStatus.SUCCESS)

        return self._record(NodeStatus.RUNNING)

    def reset(self) -> None:
        super().reset()
        self._child.reset()


# =================================================================
# BEHAVIOR TREE
# =================================================================

class BehaviorTree:
    """
    The complete behavior tree evaluated by the brain every cycle.
    Builds the priority-based reactive tree and ticks it with
    a fresh BehaviorContext populated from subsystem state.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._root: Optional[BehaviorNode] = None
        self._context: BehaviorContext = BehaviorContext()
        self._tick_count: int = 0
        self._active_behavior: str = "none"
        self._last_tick_time: float = 0.0
        self._last_tick_duration_ms: float = 0.0

        self._battery_warn_pct: float = config.get("safety", "battery_warn_pct", default=25.0)
        self._battery_critical_pct: float = config.get("safety", "battery_critical_pct", default=15.0)
        self._thermal_critical_c: float = config.get("safety", "thermal_critical_c", default=80.0)
        self._threat_confidence_min: float = config.get("behavior", "threat_confidence_min", default=0.7)
        self._investigation_confidence_min: float = config.get("behavior", "investigation_confidence_min", default=0.6)
        self._investigation_cooldown_s: float = config.get("behavior", "investigation_cooldown_s", default=30.0)

        self._on_behavior_change: Optional[Callable[[str, str], None]] = None

        logger.info("Behavior tree initialized")

    def set_root(self, root: BehaviorNode) -> None:
        """Set a custom root node (for testing or advanced configuration)."""
        self._root = root

    def build_default_tree(self) -> None:
        """
        Build the standard Cerberus behavior tree.
        Priority order enforced by PrioritySelector — safety always evaluated first.
        """
        self._root = PrioritySelector("CerberusRoot", [
            # P1: Emergency Stop
            Sequence("EmergencyStop", [
                Condition("SafetyViolation", lambda ctx: ctx.safety_violation),
                Action("ExecuteEmergencyStop", self._action_emergency_stop)
            ]),

            # P2: Return to Base
            Sequence("ReturnToBase", [
                Selector("RTBTrigger", [
                    Condition("BatteryCritical", lambda ctx: ctx.battery_pct <= self._battery_critical_pct),
                    Condition("ThermalCritical", lambda ctx: ctx.cpu_temp_c >= self._thermal_critical_c),
                    Condition("GPSLost", lambda ctx: not ctx.gps_fix and ctx.mission_active),
                    Condition("RTBRequested", lambda ctx: ctx.requesting_rtb),
                ]),
                Action("ExecuteRTB", self._action_return_to_base)
            ]),

            # P3: Threat Response
            Cooldown(
                Sequence("ThreatResponse", [
                    Condition("ThreatDetected", lambda ctx: (
                        ctx.threat_detected and
                        ctx.detection_confidence >= self._threat_confidence_min
                    )),
                    Action("RespondToThreat", self._action_threat_response)
                ]),
                cooldown_seconds=10.0,
                name="ThreatCooldown"
            ),

            # P4: Obstacle Avoidance
            Sequence("ObstacleAvoidance", [
                Condition("ObstacleDetected", lambda ctx: not ctx.obstacle_path_clear),
                Action("AvoidObstacle", self._action_avoid_obstacle)
            ]),

            # P5: Active Investigation
            Cooldown(
                Sequence("ActiveInvestigation", [
                    Condition("InterestingDetection", lambda ctx: (
                        ctx.active_detection and
                        not ctx.threat_detected and
                        ctx.detection_confidence >= self._investigation_confidence_min
                    )),
                    Action("Investigate", self._action_investigate)
                ]),
                cooldown_seconds=self._investigation_cooldown_s,
                name="InvestigationCooldown"
            ),

            # P6: Mission Execution
            Sequence("MissionExecution", [
                Condition("MissionActive", lambda ctx: ctx.mission_active and not ctx.mission_paused),
                Action("ExecuteMission", self._action_execute_mission)
            ]),

            # P7: Idle Patrol
            Action("IdlePatrol", self._action_idle_patrol)
        ])

        logger.info("Default behavior tree built — 7 priority levels")

    def tick(self, context: BehaviorContext) -> NodeStatus:
        """
        Evaluate the behavior tree with the current context.
        Called by the brain every main loop cycle.
        """
        if self._root is None:
            return NodeStatus.FAILURE

        now: float = time.time()
        context.tick_count = self._tick_count
        context.timestamp = now
        context.delta_time = now - self._last_tick_time if self._last_tick_time > 0 else 0.0

        self._context = context
        self._tick_count += 1

        tick_start: float = time.time()
        status: NodeStatus = self._root.tick(context)
        self._last_tick_duration_ms = (time.time() - tick_start) * 1000.0
        self._last_tick_time = now

        return status

    def _update_active_behavior(self, behavior_name: str) -> None:
        """Track and log behavior changes."""
        if behavior_name != self._active_behavior:
            old: str = self._active_behavior
            self._active_behavior = behavior_name
            logger.info("Behavior change: %s -> %s", old, behavior_name)
            if self._on_behavior_change is not None:
                try:
                    self._on_behavior_change(old, behavior_name)
                except Exception as e:
                    logger.error("Behavior change callback error: %s", e)

    # =================================================================
    # BEHAVIOR ACTIONS
    # =================================================================

    def _action_emergency_stop(self, ctx: BehaviorContext) -> NodeStatus:
        """P1: Kill all motors immediately. Safety overrides everything."""
        self._update_active_behavior("emergency_stop")
        ctx.requesting_stop = True
        logger.critical("EMERGENCY STOP — reason: %s", ctx.safety_reason)
        return NodeStatus.SUCCESS

    def _action_return_to_base(self, ctx: BehaviorContext) -> NodeStatus:
        """P2: Navigate home. Persists as RUNNING until home is reached."""
        self._update_active_behavior("return_to_base")
        ctx.requesting_rtb = True

        if ctx.at_home:
            logger.info("RTB complete — rover is home")
            return NodeStatus.SUCCESS

        reason: str = "unknown"
        if ctx.battery_pct <= self._battery_critical_pct:
            reason = f"battery critical ({ctx.battery_pct:.0f}%)"
        elif ctx.cpu_temp_c >= self._thermal_critical_c:
            reason = f"thermal critical ({ctx.cpu_temp_c:.1f}C)"
        elif not ctx.gps_fix:
            reason = "GPS fix lost"

        logger.warning("RTB in progress — reason: %s, distance: %.1fm", reason, ctx.distance_to_home_m)
        return NodeStatus.RUNNING

    def _action_threat_response(self, ctx: BehaviorContext) -> NodeStatus:
        """P3: React to a detected threat — alert, track, log evidence."""
        self._update_active_behavior("threat_response")

        ctx.blackboard["last_threat_type"] = ctx.detection_label
        ctx.blackboard["last_threat_confidence"] = ctx.detection_confidence
        ctx.blackboard["last_threat_time"] = ctx.timestamp

        logger.warning(
            "THREAT RESPONSE — %s (%.0f%%) at (%.6f, %.6f)",
            ctx.detection_label,
            ctx.detection_confidence * 100,
            ctx.gps_lat,
            ctx.gps_lon
        )
        return NodeStatus.SUCCESS

    def _action_avoid_obstacle(self, ctx: BehaviorContext) -> NodeStatus:
        """P4: React to obstacle detection — stop, reroute, or reverse."""
        self._update_active_behavior("obstacle_avoidance")

        if ctx.obstacle_zone == "critical":
            ctx.requesting_stop = True
            logger.warning("Obstacle CRITICAL (%.0fcm) — emergency stop", ctx.obstacle_closest_cm)
            return NodeStatus.SUCCESS

        if ctx.obstacle_zone == "caution":
            logger.info(
                "Obstacle avoidance — direction: %s, closest: %.0fcm",
                ctx.avoidance_direction, ctx.obstacle_closest_cm
            )
            return NodeStatus.RUNNING

        return NodeStatus.SUCCESS

    def _action_investigate(self, ctx: BehaviorContext) -> NodeStatus:
        """P5: Pause route and investigate an interesting detection."""
        self._update_active_behavior("investigation")

        ctx.requesting_investigation = True
        ctx.investigation_lat = ctx.gps_lat
        ctx.investigation_lon = ctx.gps_lon

        ctx.blackboard["investigation_type"] = ctx.detection_type
        ctx.blackboard["investigation_label"] = ctx.detection_label
        ctx.blackboard["investigation_confidence"] = ctx.detection_confidence

        logger.info(
            "INVESTIGATING — %s/%s (%.0f%%) at (%.6f, %.6f)",
            ctx.detection_type,
            ctx.detection_label,
            ctx.detection_confidence * 100,
            ctx.gps_lat,
            ctx.gps_lon
        )
        return NodeStatus.SUCCESS

    def _action_execute_mission(self, ctx: BehaviorContext) -> NodeStatus:
        """P6: Continue executing the active mission. Always RUNNING until mission ends."""
        self._update_active_behavior("mission_execution")
        return NodeStatus.RUNNING

    def _action_idle_patrol(self, ctx: BehaviorContext) -> NodeStatus:
        """P7: No mission loaded. Slow patrol near home or hold position."""
        self._update_active_behavior("idle_patrol")

        if not ctx.gps_fix:
            return NodeStatus.RUNNING

        if ctx.distance_to_home_m > 10.0:
            ctx.requesting_rtb = True
            return NodeStatus.RUNNING

        return NodeStatus.RUNNING

    # =================================================================
    # CALLBACKS AND STATE
    # =================================================================

    def set_behavior_change_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for behavior transitions. Signature: callback(old_name, new_name)."""
        self._on_behavior_change = callback

    @property
    def active_behavior(self) -> str:
        return self._active_behavior

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def last_tick_duration_ms(self) -> float:
        return self._last_tick_duration_ms

    @property
    def context(self) -> BehaviorContext:
        return self._context

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tick_count": self._tick_count,
            "active_behavior": self._active_behavior,
            "last_tick_ms": round(self._last_tick_duration_ms, 2)
        }
        if self._root is not None:
            result["root"] = self._root.stats()
        return result

    def get_tree_structure(self, node: Optional[BehaviorNode] = None, depth: int = 0) -> list[dict[str, Any]]:
        """Get a flat list describing the tree structure for debugging."""
        if node is None:
            node = self._root
        if node is None:
            return []

        entry: dict[str, Any] = {
            "depth": depth,
            "name": node.name,
            "type": node.__class__.__name__,
            "status": node.status.value
        }
        result: list[dict[str, Any]] = [entry]

        children: Optional[list[BehaviorNode]] = None
        if isinstance(node, (Selector, Sequence, PrioritySelector)):
            children = node._children
        elif isinstance(node, (Inverter, Cooldown, Retry, RepeatUntilFail)):
            children = [node._child]

        if children:
            for child in children:
                result.extend(self.get_tree_structure(child, depth + 1))

        return result