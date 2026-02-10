"""
Tests for cerberus.intelligence.behavior_tree — BehaviorTree
Validates node types (Selector, Sequence, PrioritySelector), leaf nodes
(Condition, Action), decorators (Inverter, Cooldown, Retry, RepeatUntilFail),
context population, behavior priority, preemption, and the default tree.
"""

import time
import pytest
from typing import Any
from unittest.mock import MagicMock

from cerberus.core.config import CerberusConfig
from cerberus.intelligence.behavior_tree import (
    BehaviorTree, BehaviorContext, BehaviorNode, NodeStatus,
    BehaviorPriority, Selector, Sequence, PrioritySelector,
    Condition, Action, Inverter, Cooldown, Retry, RepeatUntilFail
)


def success_action(ctx: BehaviorContext) -> NodeStatus:
    return NodeStatus.SUCCESS


def failure_action(ctx: BehaviorContext) -> NodeStatus:
    return NodeStatus.FAILURE


def running_action(ctx: BehaviorContext) -> NodeStatus:
    return NodeStatus.RUNNING


class TestNodeStatus:
    """NodeStatus enum."""

    def test_all_statuses(self) -> None:
        assert NodeStatus.SUCCESS.value == "success"
        assert NodeStatus.FAILURE.value == "failure"
        assert NodeStatus.RUNNING.value == "running"


class TestBehaviorPriority:
    """BehaviorPriority ordering."""

    def test_emergency_highest(self) -> None:
        assert BehaviorPriority.EMERGENCY_STOP.value < BehaviorPriority.RETURN_TO_BASE.value

    def test_idle_lowest(self) -> None:
        assert BehaviorPriority.IDLE_PATROL.value > BehaviorPriority.MISSION_EXECUTION.value

    def test_full_ordering(self) -> None:
        ordered: list[int] = [p.value for p in BehaviorPriority]
        assert ordered == sorted(ordered)


class TestBehaviorContext:
    """BehaviorContext shared state."""

    def test_default_context(self) -> None:
        ctx: BehaviorContext = BehaviorContext()
        assert ctx.battery_pct == 100.0
        assert ctx.safety_violation is False
        assert ctx.obstacle_path_clear is True
        assert ctx.mission_active is False
        assert ctx.requesting_rtb is False
        assert ctx.requesting_stop is False

    def test_blackboard(self) -> None:
        ctx: BehaviorContext = BehaviorContext()
        ctx.blackboard["key"] = "value"
        assert ctx.blackboard["key"] == "value"

    def test_to_dict(self) -> None:
        ctx: BehaviorContext = BehaviorContext()
        d: dict[str, Any] = ctx.to_dict()
        assert "tick" in d
        assert "battery_pct" in d
        assert "mission_active" in d
        assert "requesting_stop" in d


class TestConditionNode:
    """Condition leaf node."""

    def test_true_predicate_succeeds(self) -> None:
        node: Condition = Condition("IsTrue", lambda ctx: True)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_false_predicate_fails(self) -> None:
        node: Condition = Condition("IsFalse", lambda ctx: False)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_predicate_reads_context(self) -> None:
        node: Condition = Condition("LowBattery", lambda ctx: ctx.battery_pct < 20.0)
        ctx: BehaviorContext = BehaviorContext(battery_pct=15.0)
        assert node.tick(ctx) == NodeStatus.SUCCESS

        ctx.battery_pct = 80.0
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_exception_returns_failure(self) -> None:
        def bad_predicate(ctx: BehaviorContext) -> bool:
            raise RuntimeError("boom")

        node: Condition = Condition("BadPred", bad_predicate)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_stats_tracked(self) -> None:
        node: Condition = Condition("Counter", lambda ctx: True)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        node.tick(ctx)
        s: dict[str, Any] = node.stats()
        assert s["runs"] == 2
        assert s["successes"] == 2


class TestActionNode:
    """Action leaf node."""

    def test_success_action(self) -> None:
        node: Action = Action("Succeed", success_action)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_failure_action(self) -> None:
        node: Action = Action("Fail", failure_action)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_running_action(self) -> None:
        node: Action = Action("Running", running_action)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.RUNNING

    def test_action_modifies_context(self) -> None:
        def set_flag(ctx: BehaviorContext) -> NodeStatus:
            ctx.requesting_stop = True
            return NodeStatus.SUCCESS

        node: Action = Action("SetFlag", set_flag)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        assert ctx.requesting_stop is True

    def test_exception_returns_failure(self) -> None:
        def bad_action(ctx: BehaviorContext) -> NodeStatus:
            raise RuntimeError("crash")

        node: Action = Action("BadAction", bad_action)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE


class TestSelector:
    """Selector composite (OR node)."""

    def test_first_success_wins(self) -> None:
        node: Selector = Selector("OrNode", [
            Action("A", failure_action),
            Action("B", success_action),
            Action("C", failure_action),
        ])
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_all_fail(self) -> None:
        node: Selector = Selector("AllFail", [
            Action("A", failure_action),
            Action("B", failure_action),
        ])
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_running_stops_evaluation(self) -> None:
        call_log: list[str] = []

        def log_and_fail(ctx: BehaviorContext) -> NodeStatus:
            call_log.append("fail")
            return NodeStatus.FAILURE

        def log_and_run(ctx: BehaviorContext) -> NodeStatus:
            call_log.append("run")
            return NodeStatus.RUNNING

        def log_and_succeed(ctx: BehaviorContext) -> NodeStatus:
            call_log.append("succeed")
            return NodeStatus.SUCCESS

        node: Selector = Selector("RunTest", [
            Action("A", log_and_fail),
            Action("B", log_and_run),
            Action("C", log_and_succeed),
        ])
        ctx: BehaviorContext = BehaviorContext()
        result: NodeStatus = node.tick(ctx)
        assert result == NodeStatus.RUNNING
        assert "succeed" not in call_log

    def test_empty_selector_fails(self) -> None:
        node: Selector = Selector("Empty")
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_add_child(self) -> None:
        node: Selector = Selector("Build")
        node.add_child(Action("A", success_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_reset(self) -> None:
        node: Selector = Selector("Resettable", [
            Action("A", success_action)
        ])
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        assert node.status == NodeStatus.SUCCESS
        node.reset()
        assert node.status == NodeStatus.FAILURE


class TestSequence:
    """Sequence composite (AND node)."""

    def test_all_succeed(self) -> None:
        node: Sequence = Sequence("AndNode", [
            Action("A", success_action),
            Action("B", success_action),
            Action("C", success_action),
        ])
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_first_failure_stops(self) -> None:
        call_log: list[str] = []

        def log_succeed(ctx: BehaviorContext) -> NodeStatus:
            call_log.append("succeed")
            return NodeStatus.SUCCESS

        def log_fail(ctx: BehaviorContext) -> NodeStatus:
            call_log.append("fail")
            return NodeStatus.FAILURE

        node: Sequence = Sequence("FailTest", [
            Action("A", log_succeed),
            Action("B", log_fail),
            Action("C", log_succeed),
        ])
        ctx: BehaviorContext = BehaviorContext()
        result: NodeStatus = node.tick(ctx)
        assert result == NodeStatus.FAILURE
        assert len(call_log) == 2

    def test_running_stops_evaluation(self) -> None:
        node: Sequence = Sequence("RunSeq", [
            Action("A", success_action),
            Action("B", running_action),
            Action("C", success_action),
        ])
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.RUNNING

    def test_empty_sequence_succeeds(self) -> None:
        node: Sequence = Sequence("Empty")
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_add_child(self) -> None:
        node: Sequence = Sequence("Build")
        node.add_child(Action("A", success_action))
        node.add_child(Action("B", success_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS


class TestPrioritySelector:
    """PrioritySelector with preemption."""

    def test_highest_priority_wins(self) -> None:
        node: PrioritySelector = PrioritySelector("Priority", [
            Sequence("P1", [
                Condition("Emergency", lambda ctx: ctx.safety_violation),
                Action("Stop", success_action)
            ]),
            Action("P2_Patrol", running_action),
        ])

        ctx: BehaviorContext = BehaviorContext(safety_violation=False)
        result: NodeStatus = node.tick(ctx)
        assert result == NodeStatus.RUNNING

        ctx.safety_violation = True
        result = node.tick(ctx)
        assert result == NodeStatus.SUCCESS

    def test_preemption_resets_lower(self) -> None:
        lower_reset: list[bool] = [False]

        class TrackingAction(BehaviorNode):
            def tick(self, context: BehaviorContext) -> NodeStatus:
                return self._record(NodeStatus.RUNNING)
            def reset(self) -> None:
                super().reset()
                lower_reset[0] = True

        node: PrioritySelector = PrioritySelector("Preempt", [
            Sequence("P1", [
                Condition("Trigger", lambda ctx: ctx.safety_violation),
                Action("Stop", success_action)
            ]),
            TrackingAction("LowerTask"),
        ])

        ctx: BehaviorContext = BehaviorContext(safety_violation=False)
        node.tick(ctx)
        assert node._running_child_idx == 1

        ctx.safety_violation = True
        node.tick(ctx)
        assert lower_reset[0] is True

    def test_all_fail(self) -> None:
        node: PrioritySelector = PrioritySelector("AllFail", [
            Action("A", failure_action),
            Action("B", failure_action),
        ])
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE
        assert node._running_child_idx == -1


class TestInverter:
    """Inverter decorator."""

    def test_inverts_success(self) -> None:
        node: Inverter = Inverter(Action("A", success_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_inverts_failure(self) -> None:
        node: Inverter = Inverter(Action("A", failure_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_running_passes_through(self) -> None:
        node: Inverter = Inverter(Action("A", running_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.RUNNING

    def test_name_auto_generated(self) -> None:
        inner: Action = Action("MyAction", success_action)
        node: Inverter = Inverter(inner)
        assert "MyAction" in node.name


class TestCooldown:
    """Cooldown decorator."""

    def test_first_call_succeeds(self) -> None:
        node: Cooldown = Cooldown(Action("A", success_action), cooldown_seconds=1.0)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_second_call_blocked(self) -> None:
        node: Cooldown = Cooldown(Action("A", success_action), cooldown_seconds=10.0)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        assert node.tick(ctx) == NodeStatus.FAILURE

    def test_cooldown_expires(self) -> None:
        node: Cooldown = Cooldown(Action("A", success_action), cooldown_seconds=0.1)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        time.sleep(0.15)
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_failure_does_not_trigger_cooldown(self) -> None:
        node: Cooldown = Cooldown(Action("A", failure_action), cooldown_seconds=10.0)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        assert node.tick(ctx) == NodeStatus.FAILURE


class TestRetry:
    """Retry decorator."""

    def test_succeeds_immediately(self) -> None:
        node: Retry = Retry(Action("A", success_action), max_attempts=3)
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_retries_on_failure(self) -> None:
        attempt: list[int] = [0]

        def succeed_on_third(ctx: BehaviorContext) -> NodeStatus:
            attempt[0] += 1
            if attempt[0] >= 3:
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        node: Retry = Retry(Action("A", succeed_on_third), max_attempts=5)
        ctx: BehaviorContext = BehaviorContext()

        result1: NodeStatus = node.tick(ctx)
        assert result1 == NodeStatus.RUNNING

        result2: NodeStatus = node.tick(ctx)
        assert result2 == NodeStatus.RUNNING

        result3: NodeStatus = node.tick(ctx)
        assert result3 == NodeStatus.SUCCESS

    def test_max_attempts_exceeded(self) -> None:
        node: Retry = Retry(Action("A", failure_action), max_attempts=2)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        result: NodeStatus = node.tick(ctx)
        assert result == NodeStatus.FAILURE

    def test_reset_clears_attempts(self) -> None:
        node: Retry = Retry(Action("A", failure_action), max_attempts=3)
        ctx: BehaviorContext = BehaviorContext()
        node.tick(ctx)
        node.tick(ctx)
        node.reset()
        assert node._attempts == 0


class TestRepeatUntilFail:
    """RepeatUntilFail decorator."""

    def test_running_while_child_succeeds(self) -> None:
        node: RepeatUntilFail = RepeatUntilFail(Action("A", success_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.RUNNING

    def test_success_when_child_fails(self) -> None:
        node: RepeatUntilFail = RepeatUntilFail(Action("A", failure_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.SUCCESS

    def test_running_passes_through(self) -> None:
        node: RepeatUntilFail = RepeatUntilFail(Action("A", running_action))
        ctx: BehaviorContext = BehaviorContext()
        assert node.tick(ctx) == NodeStatus.RUNNING


class TestBehaviorTree:
    """Full BehaviorTree with default tree."""

    def test_build_default_tree(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        assert tree._root is not None

    def test_tick_increments_counter(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext()
        tree.tick(ctx)
        tree.tick(ctx)
        assert tree.tick_count == 2

    def test_tick_measures_duration(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext()
        tree.tick(ctx)
        assert tree.last_tick_duration_ms >= 0

    def test_idle_patrol_default_behavior(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            gps_fix=True, at_home=True, distance_to_home_m=0.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "idle_patrol"

    def test_emergency_stop_preempts_all(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        ctx: BehaviorContext = BehaviorContext(
            mission_active=True, gps_fix=True
        )
        tree.tick(ctx)
        assert tree.active_behavior == "mission_execution"

        ctx.safety_violation = True
        ctx.safety_reason = "motor overcurrent"
        tree.tick(ctx)
        assert tree.active_behavior == "emergency_stop"
        assert ctx.requesting_stop is True

    def test_battery_critical_triggers_rtb(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            battery_pct=10.0, gps_fix=True, distance_to_home_m=50.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "return_to_base"
        assert ctx.requesting_rtb is True

    def test_thermal_critical_triggers_rtb(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            cpu_temp_c=85.0, gps_fix=True, distance_to_home_m=50.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "return_to_base"

    def test_gps_lost_during_mission_triggers_rtb(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            gps_fix=False, mission_active=True, distance_to_home_m=50.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "return_to_base"

    def test_rtb_completes_at_home(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            battery_pct=10.0, gps_fix=True, at_home=True, distance_to_home_m=0.0
        )
        status: NodeStatus = tree.tick(ctx)
        assert tree.active_behavior == "return_to_base"

    def test_obstacle_avoidance_triggers(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            obstacle_path_clear=False,
            obstacle_zone="caution",
            obstacle_closest_cm=40.0,
            avoidance_direction="left",
            gps_fix=True
        )
        tree.tick(ctx)
        assert tree.active_behavior == "obstacle_avoidance"

    def test_critical_obstacle_requests_stop(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            obstacle_path_clear=False,
            obstacle_zone="critical",
            obstacle_closest_cm=10.0,
            gps_fix=True
        )
        tree.tick(ctx)
        assert ctx.requesting_stop is True

    def test_threat_response_triggers(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            threat_detected=True,
            detection_confidence=0.9,
            detection_label="intruder",
            gps_fix=True,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        tree.tick(ctx)
        assert tree.active_behavior == "threat_response"
        assert ctx.blackboard.get("last_threat_type") == "intruder"

    def test_investigation_triggers(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            active_detection=True,
            threat_detected=False,
            detection_confidence=0.75,
            detection_type="weed",
            detection_label="dandelion",
            gps_fix=True,
            gps_lat=36.17,
            gps_lon=-115.14
        )
        tree.tick(ctx)
        assert tree.active_behavior == "investigation"
        assert ctx.requesting_investigation is True

    def test_mission_execution(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            mission_active=True, mission_paused=False, gps_fix=True
        )
        tree.tick(ctx)
        assert tree.active_behavior == "mission_execution"

    def test_paused_mission_falls_to_idle(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(
            mission_active=True, mission_paused=True,
            gps_fix=True, at_home=True, distance_to_home_m=0.0
        )
        tree.tick(ctx)
        assert tree.active_behavior == "idle_patrol"


class TestBehaviorChangeCallback:
    """Behavior transition notifications."""

    def test_callback_fires(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        transitions: list[tuple[str, str]] = []

        def on_change(old: str, new: str) -> None:
            transitions.append((old, new))

        tree.set_behavior_change_callback(on_change)

        ctx: BehaviorContext = BehaviorContext(gps_fix=True, at_home=True)
        tree.tick(ctx)
        assert len(transitions) >= 1

    def test_callback_error_handled(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        def bad_callback(old: str, new: str) -> None:
            raise RuntimeError("callback crash")

        tree.set_behavior_change_callback(bad_callback)

        ctx: BehaviorContext = BehaviorContext(gps_fix=True, at_home=True)
        tree.tick(ctx)


class TestCustomTree:
    """Custom tree with set_root."""

    def test_set_custom_root(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        custom: Action = Action("Custom", success_action)
        tree.set_root(custom)
        ctx: BehaviorContext = BehaviorContext()
        result: NodeStatus = tree.tick(ctx)
        assert result == NodeStatus.SUCCESS

    def test_no_root_returns_failure(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        ctx: BehaviorContext = BehaviorContext()
        result: NodeStatus = tree.tick(ctx)
        assert result == NodeStatus.FAILURE


class TestTreeIntrospection:
    """Stats and tree structure."""

    def test_stats(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(gps_fix=True, at_home=True)
        tree.tick(ctx)

        s: dict[str, Any] = tree.stats()
        assert s["tick_count"] == 1
        assert "active_behavior" in s
        assert "root" in s

    def test_get_tree_structure(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()

        structure: list[dict[str, Any]] = tree.get_tree_structure()
        assert len(structure) > 0
        assert structure[0]["name"] == "CerberusRoot"
        assert structure[0]["depth"] == 0

        has_children: bool = any(s["depth"] > 0 for s in structure)
        assert has_children is True

    def test_tree_structure_empty(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        structure: list[dict[str, Any]] = tree.get_tree_structure()
        assert len(structure) == 0

    def test_context_property(self, config: CerberusConfig) -> None:
        tree: BehaviorTree = BehaviorTree(config)
        tree.build_default_tree()
        ctx: BehaviorContext = BehaviorContext(battery_pct=42.0)
        tree.tick(ctx)
        assert tree.context.battery_pct == 42.0