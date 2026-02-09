"""
Cerberus Safety Watchdog
The most critical module in the system. Monitors thermal, battery,
power, and motor conditions. Has absolute authority to override any
mission and force RTB or emergency shutdown. Safety outranks autonomy.
This module cannot be disabled.
"""

import time
import threading
import logging
from typing import Any, Optional, Callable
from enum import Enum

from cerberus.core.config import CerberusConfig
from cerberus.core.health import HealthMonitor, HealthSnapshot


logger: logging.Logger = logging.getLogger(__name__)


class SafetyState(Enum):
    """Cerberus safety states — ordered by severity."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class SafetyAction(Enum):
    """Actions the safety watchdog can command."""
    NONE = "none"
    WARN = "warn"
    THROTTLE_MOTORS = "throttle_motors"
    STOP_MOTORS = "stop_motors"
    FORCE_RTB = "force_rtb"
    ABORT_MISSION = "abort_mission"
    SAFE_SHUTDOWN = "safe_shutdown"
    EMERGENCY_STOP = "emergency_stop"


class SafetyViolation:
    """Represents a detected safety condition."""

    def __init__(
        self,
        source: str,
        message: str,
        action: SafetyAction,
        state: SafetyState
    ) -> None:
        self.source: str = source
        self.message: str = message
        self.action: SafetyAction = action
        self.state: SafetyState = state
        self.timestamp: float = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "action": self.action.value,
            "state": self.state.value,
            "timestamp": self.timestamp
        }

    def __repr__(self) -> str:
        return f"SafetyViolation({self.source}: {self.message} → {self.action.value})"


class SafetyWatchdog:
    """
    Background safety watchdog for Cerberus.
    Continuously evaluates system health against safety thresholds.
    Cannot be disabled. Has authority to override all other subsystems.
    """

    def __init__(
        self,
        config: Optional[CerberusConfig] = None,
        health_monitor: Optional[HealthMonitor] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._health: Optional[HealthMonitor] = health_monitor

        self._thermal_warn: float = config.get("safety", "thermal_warning_c", default=70)
        self._thermal_shutdown: float = config.get("safety", "thermal_shutdown_c", default=80)
        self._battery_warn: float = config.get("safety", "battery_warning_pct", default=25)
        self._battery_critical: float = config.get("safety", "battery_critical_pct", default=15)
        self._battery_shutdown: float = config.get("safety", "battery_shutdown_pct", default=8)
        self._interval: int = config.get("safety", "watchdog_interval_seconds", default=5)
        self._max_motor_runtime: int = config.get("safety", "max_motor_runtime_seconds", default=300)
        self._ambient_limit: float = config.get("safety", "ambient_temp_limit_c", default=55)

        self._pi_rail_min: float = config.get(
            "power", "channels", "pi_rail", "min_voltage", default=4.75
        )
        self._motor_overcurrent: float = config.get(
            "power", "channels", "motors", "overcurrent_shutdown_a", default=30.0
        )
        self._motor_stall: float = config.get(
            "power", "channels", "motors", "stall_current_a", default=25.0
        )

        self._state: SafetyState = SafetyState.NORMAL
        self._violations: list[SafetyViolation] = []
        self._motor_start_time: Optional[float] = None
        self._motors_active: bool = False
        self._shutdown_requested: bool = False

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()

        self._action_callbacks: dict[SafetyAction, list[Callable[[SafetyViolation], None]]] = {
            action: [] for action in SafetyAction
        }

        self._mqtt: Optional[Any] = None
        self._db: Optional[Any] = None

        logger.info(
            "Safety watchdog created — thermal_warn=%.0f°C, thermal_shutdown=%.0f°C, "
            "battery_warn=%d%%, battery_critical=%d%%, battery_shutdown=%d%%",
            self._thermal_warn, self._thermal_shutdown,
            self._battery_warn, self._battery_critical, self._battery_shutdown
        )

    def bind_health(self, health_monitor: HealthMonitor) -> None:
        """Bind the health monitor. Called during system init."""
        self._health = health_monitor

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for alert publishing."""
        self._mqtt = mqtt_client

    def bind_db(self, db: Any) -> None:
        """Bind database for violation logging."""
        self._db = db

    def register_action(
        self,
        action: SafetyAction,
        callback: Callable[[SafetyViolation], None]
    ) -> None:
        """
        Register a callback for a safety action.
        The motor driver registers for STOP_MOTORS and THROTTLE_MOTORS.
        The mission planner registers for FORCE_RTB and ABORT_MISSION.
        The brain registers for SAFE_SHUTDOWN.
        """
        self._action_callbacks[action].append(callback)
        logger.info("Safety action registered: %s", action.value)

    def notify_motors_active(self, active: bool) -> None:
        """Called by motor controller to track motor runtime."""
        with self._lock:
            if active and not self._motors_active:
                self._motor_start_time = time.time()
            elif not active:
                self._motor_start_time = None
            self._motors_active = active

    def _evaluate(self, snapshot: HealthSnapshot) -> list[SafetyViolation]:
        """Evaluate all safety conditions against current health."""
        violations: list[SafetyViolation] = []

        violations.extend(self._check_thermal(snapshot))
        violations.extend(self._check_battery(snapshot))
        violations.extend(self._check_power_rail(snapshot))
        violations.extend(self._check_motor_current(snapshot))
        violations.extend(self._check_motor_runtime())

        return violations

    def _check_thermal(self, snapshot: HealthSnapshot) -> list[SafetyViolation]:
        """Check CPU temperature against thresholds."""
        violations: list[SafetyViolation] = []
        temp: float = snapshot.cpu_temp_c

        if temp <= 0:
            return violations

        if temp >= self._thermal_shutdown:
            violations.append(SafetyViolation(
                source="thermal",
                message=f"CPU temperature critical: {temp:.1f}°C — emergency shutdown",
                action=SafetyAction.SAFE_SHUTDOWN,
                state=SafetyState.EMERGENCY
            ))
        elif temp >= self._thermal_warn:
            violations.append(SafetyViolation(
                source="thermal",
                message=f"CPU temperature elevated: {temp:.1f}°C — throttling motors",
                action=SafetyAction.THROTTLE_MOTORS,
                state=SafetyState.WARNING
            ))

        return violations

    def _check_battery(self, snapshot: HealthSnapshot) -> list[SafetyViolation]:
        """Check battery level against thresholds."""
        violations: list[SafetyViolation] = []
        pct: float = snapshot.battery_pct
        voltage: float = snapshot.battery_voltage

        if voltage <= 0 and pct >= 100:
            return violations

        if pct <= self._battery_shutdown:
            violations.append(SafetyViolation(
                source="battery",
                message=f"Battery critically low: {pct:.1f}% ({voltage:.2f}V) — safe shutdown to protect cells",
                action=SafetyAction.SAFE_SHUTDOWN,
                state=SafetyState.EMERGENCY
            ))
        elif pct <= self._battery_critical:
            violations.append(SafetyViolation(
                source="battery",
                message=f"Battery critical: {pct:.1f}% ({voltage:.2f}V) — forcing RTB",
                action=SafetyAction.FORCE_RTB,
                state=SafetyState.CRITICAL
            ))
        elif pct <= self._battery_warn:
            violations.append(SafetyViolation(
                source="battery",
                message=f"Battery low: {pct:.1f}% ({voltage:.2f}V) — warning",
                action=SafetyAction.WARN,
                state=SafetyState.WARNING
            ))

        return violations

    def _check_power_rail(self, snapshot: HealthSnapshot) -> list[SafetyViolation]:
        """Check Pi 5V rail stability."""
        violations: list[SafetyViolation] = []
        voltage: float = snapshot.pi_rail_voltage

        if voltage <= 0:
            return violations

        if voltage < self._pi_rail_min:
            violations.append(SafetyViolation(
                source="power_rail",
                message=f"Pi 5V rail sagging: {voltage:.2f}V (min {self._pi_rail_min:.2f}V) — stopping motors",
                action=SafetyAction.STOP_MOTORS,
                state=SafetyState.CRITICAL
            ))

        return violations

    def _check_motor_current(self, snapshot: HealthSnapshot) -> list[SafetyViolation]:
        """Check motor current for overcurrent and stall conditions."""
        violations: list[SafetyViolation] = []
        current: float = snapshot.motor_current_a

        if current <= 0:
            return violations

        if current >= self._motor_overcurrent:
            violations.append(SafetyViolation(
                source="motor_current",
                message=f"Motor overcurrent: {current:.1f}A (limit {self._motor_overcurrent:.1f}A) — emergency stop",
                action=SafetyAction.EMERGENCY_STOP,
                state=SafetyState.EMERGENCY
            ))
        elif current >= self._motor_stall:
            violations.append(SafetyViolation(
                source="motor_current",
                message=f"Possible motor stall: {current:.1f}A — stopping motors",
                action=SafetyAction.STOP_MOTORS,
                state=SafetyState.CRITICAL
            ))

        return violations

    def _check_motor_runtime(self) -> list[SafetyViolation]:
        """Check if motors have been running too long continuously."""
        violations: list[SafetyViolation] = []

        with self._lock:
            if not self._motors_active or self._motor_start_time is None:
                return violations
            runtime: float = time.time() - self._motor_start_time

        if runtime >= self._max_motor_runtime:
            violations.append(SafetyViolation(
                source="motor_runtime",
                message=f"Motors running {runtime:.0f}s continuously (max {self._max_motor_runtime}s) — stopping",
                action=SafetyAction.STOP_MOTORS,
                state=SafetyState.WARNING
            ))

        return violations

    def _resolve_highest_action(
        self,
        violations: list[SafetyViolation]
    ) -> Optional[SafetyViolation]:
        """From a list of violations, return the most severe one."""
        if not violations:
            return None

        severity_order: list[SafetyAction] = [
            SafetyAction.EMERGENCY_STOP,
            SafetyAction.SAFE_SHUTDOWN,
            SafetyAction.FORCE_RTB,
            SafetyAction.ABORT_MISSION,
            SafetyAction.STOP_MOTORS,
            SafetyAction.THROTTLE_MOTORS,
            SafetyAction.WARN,
            SafetyAction.NONE,
        ]

        return min(violations, key=lambda v: severity_order.index(v.action))

    def _execute_action(self, violation: SafetyViolation) -> None:
        """Execute callbacks registered for this safety action."""
        callbacks: list[Callable] = self._action_callbacks.get(violation.action, [])

        if not callbacks:
            logger.warning(
                "No handler registered for safety action: %s — %s",
                violation.action.value, violation.message
            )
            return

        for callback in callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(
                    "Safety action callback failed for %s: %s",
                    violation.action.value, e
                )

    def _update_state(self, violations: list[SafetyViolation]) -> None:
        """Update the overall safety state based on current violations."""
        if not violations:
            with self._lock:
                if self._state != SafetyState.NORMAL:
                    logger.info("Safety state restored to NORMAL")
                self._state = SafetyState.NORMAL
                self._violations.clear()
            return

        worst: SafetyState = SafetyState.NORMAL
        state_order: list[SafetyState] = [
            SafetyState.NORMAL,
            SafetyState.WARNING,
            SafetyState.CRITICAL,
            SafetyState.EMERGENCY,
            SafetyState.SHUTDOWN,
        ]

        for v in violations:
            if state_order.index(v.state) > state_order.index(worst):
                worst = v.state

        with self._lock:
            if worst != self._state:
                logger.warning("Safety state changed: %s → %s", self._state.value, worst.value)
            self._state = worst
            self._violations = list(violations)

    def _log_violations(self, violations: list[SafetyViolation]) -> None:
        """Log violations to console, database, and MQTT."""
        for v in violations:
            if v.state == SafetyState.EMERGENCY:
                logger.critical("[SAFETY] %s", v.message)
            elif v.state == SafetyState.CRITICAL:
                logger.error("[SAFETY] %s", v.message)
            else:
                logger.warning("[SAFETY] %s", v.message)

            if self._db is not None:
                try:
                    self._db.log_system_event(
                        event_type="safety_violation",
                        source=v.source,
                        message=v.message,
                        severity=v.state.value,
                        metadata=str(v.to_dict())
                    )
                except Exception as e:
                    logger.error("Failed to log safety violation to DB: %s", e)

            if self._mqtt is not None:
                try:
                    self._mqtt.publish_alert(v.to_dict())
                except Exception:
                    pass

    def check_now(self) -> SafetyState:
        """Run an immediate safety evaluation. Returns current safety state."""
        if self._health is None:
            logger.error("Safety watchdog has no health monitor bound")
            return self._state

        snapshot: HealthSnapshot = self._health.latest
        violations: list[SafetyViolation] = self._evaluate(snapshot)

        self._update_state(violations)

        if violations:
            self._log_violations(violations)
            highest: Optional[SafetyViolation] = self._resolve_highest_action(violations)
            if highest and highest.action != SafetyAction.NONE:
                self._execute_action(highest)

        return self._state

    def _watchdog_loop(self) -> None:
        """Background watchdog loop — runs every interval, cannot be stopped externally."""
        logger.info("Safety watchdog ACTIVE — interval=%ds", self._interval)

        while self._running:
            try:
                state: SafetyState = self.check_now()

                if state == SafetyState.SHUTDOWN:
                    logger.critical("[SAFETY] Shutdown state reached — watchdog requesting system halt")
                    self._shutdown_requested = True
                    break

            except Exception as e:
                logger.error("Safety watchdog evaluation error: %s", e)

            for _ in range(self._interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("Safety watchdog stopped")

    @property
    def current_state(self) -> SafetyState:
        with self._lock:
            return self._state

    @property
    def active_violations(self) -> list[SafetyViolation]:
        with self._lock:
            return list(self._violations)

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    @property
    def is_safe_to_drive(self) -> bool:
        """Quick check for motor controllers before executing drive commands."""
        with self._lock:
            if self._state in (SafetyState.EMERGENCY, SafetyState.SHUTDOWN):
                return False
            for v in self._violations:
                if v.action in (
                    SafetyAction.EMERGENCY_STOP,
                    SafetyAction.STOP_MOTORS,
                    SafetyAction.SAFE_SHUTDOWN
                ):
                    return False
            return True

    @property
    def is_safe_for_mission(self) -> bool:
        """Quick check for mission planner before starting a mission."""
        with self._lock:
            if self._state in (SafetyState.CRITICAL, SafetyState.EMERGENCY, SafetyState.SHUTDOWN):
                return False
            return True

    def start(self) -> None:
        """Start the safety watchdog background thread."""
        if self._running:
            logger.warning("Safety watchdog already running")
            return

        if self._health is None:
            raise RuntimeError("Cannot start safety watchdog without health monitor")

        self._running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="safety-watchdog",
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the safety watchdog. Only called during system shutdown."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            if self._thread.is_alive():
                logger.warning("Safety watchdog thread did not stop cleanly")
            self._thread = None

        logger.info("Safety watchdog shutdown complete")

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        return (
            f"SafetyWatchdog(state={self._state.value}, "
            f"violations={len(self._violations)}, "
            f"running={self._running})"
        )