"""
Cerberus 6WD Skid-Steer Drive Controller
Translates high-level drive commands into left/right motor speeds.
Handles forward, reverse, turning, spinning, and arc driving.
Integrates with safety watchdog for motor protection.
"""

import time
import logging
import threading
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

from cerberus.core.config import CerberusConfig
from cerberus.mobility.motor_driver import MotorDriver


logger: logging.Logger = logging.getLogger(__name__)


class DriveMode(Enum):
    """Current drive mode."""
    STOPPED = "stopped"
    FORWARD = "forward"
    REVERSE = "reverse"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SPIN_LEFT = "spin_left"
    SPIN_RIGHT = "spin_right"
    ARC = "arc"


@dataclass
class DriveState:
    """Snapshot of current drive system state."""
    mode: DriveMode = DriveMode.STOPPED
    speed: float = 0.0
    turn: float = 0.0
    left_speed: float = 0.0
    right_speed: float = 0.0
    throttle_limit: float = 1.0
    enabled: bool = True
    safe_to_drive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "left_speed": round(self.left_speed, 3),
            "right_speed": round(self.right_speed, 3),
            "throttle_limit": round(self.throttle_limit, 3),
            "enabled": self.enabled,
            "safe_to_drive": self.safe_to_drive
        }


class DriveController:
    """
    6WD skid-steer drive controller.
    Takes speed (-1.0 to 1.0) and turn (-1.0 to 1.0) inputs
    and converts them to left/right motor commands.

    Turn mixing:
        turn = 0.0  → straight
        turn = -1.0 → full left (spin in place if speed=0)
        turn = +1.0 → full right (spin in place if speed=0)

    Throttle limiting:
        Safety watchdog can reduce max speed via throttle_limit.
        All output speeds are multiplied by this value.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config

        left_cfg: dict[str, Any] = config.section("motors")["left"]
        right_cfg: dict[str, Any] = config.section("motors")["right"]

        self._left_motor: MotorDriver = MotorDriver(
            name="left",
            pwm_pin=left_cfg["pwm_pin"],
            forward_pin=left_cfg["forward_pin"],
            reverse_pin=left_cfg["reverse_pin"],
            config=config
        )

        self._right_motor: MotorDriver = MotorDriver(
            name="right",
            pwm_pin=right_cfg["pwm_pin"],
            forward_pin=right_cfg["forward_pin"],
            reverse_pin=right_cfg["reverse_pin"],
            config=config
        )

        self._max_duty: float = config.get("motors", "max_duty_cycle", default=1.0)
        self._deadband: float = config.get("motors", "deadband", default=0.1)

        self._throttle_limit: float = 1.0
        self._enabled: bool = True
        self._safe_to_drive: bool = True
        self._lock: threading.Lock = threading.Lock()

        self._current_speed: float = 0.0
        self._current_turn: float = 0.0
        self._mode: DriveMode = DriveMode.STOPPED
        self._last_drive_time: Optional[float] = None

        self._safety_callback: Optional[Any] = None

        logger.info("Drive controller initialized — 6WD skid-steer")

    def drive(self, speed: float, turn: float = 0.0) -> None:
        """
        Main drive command.
        speed: -1.0 (full reverse) to 1.0 (full forward)
        turn:  -1.0 (full left) to 1.0 (full right)
        """
        with self._lock:
            if not self._enabled:
                logger.warning("Drive command ignored — controller disabled")
                return
            if not self._safe_to_drive:
                logger.warning("Drive command ignored — safety violation active")
                return

        speed = max(-1.0, min(1.0, speed))
        turn = max(-1.0, min(1.0, turn))

        with self._lock:
            throttle: float = self._throttle_limit
        speed *= throttle

        left_speed: float
        right_speed: float
        left_speed, right_speed = self._mix(speed, turn)

        mode: DriveMode = self._determine_mode(speed, turn, left_speed, right_speed)

        with self._lock:
            self._current_speed = speed
            self._current_turn = turn
            self._mode = mode
            self._last_drive_time = time.time()

        self._left_motor.set_speed(left_speed)
        self._right_motor.set_speed(right_speed)

        logger.debug(
            "Drive: speed=%.2f, turn=%.2f → L=%.2f, R=%.2f (%s)",
            speed, turn, left_speed, right_speed, mode.value
        )

    def _mix(self, speed: float, turn: float) -> tuple[float, float]:
        """
        Mix speed and turn into left/right motor values.
        Uses differential steering: turn reduces one side proportionally.
        """
        if abs(speed) < self._deadband and abs(turn) >= self._deadband:
            left: float = -turn
            right: float = turn
            return (
                max(-self._max_duty, min(self._max_duty, left)),
                max(-self._max_duty, min(self._max_duty, right))
            )

        left = speed + (turn * abs(speed))
        right = speed - (turn * abs(speed))

        max_magnitude: float = max(abs(left), abs(right))
        if max_magnitude > self._max_duty:
            scale: float = self._max_duty / max_magnitude
            left *= scale
            right *= scale

        return (
            max(-self._max_duty, min(self._max_duty, left)),
            max(-self._max_duty, min(self._max_duty, right))
        )

    def _determine_mode(
        self,
        speed: float,
        turn: float,
        left: float,
        right: float
    ) -> DriveMode:
        """Determine the current drive mode from inputs."""
        if abs(left) < self._deadband and abs(right) < self._deadband:
            return DriveMode.STOPPED

        if abs(speed) < self._deadband:
            if turn < -self._deadband:
                return DriveMode.SPIN_LEFT
            elif turn > self._deadband:
                return DriveMode.SPIN_RIGHT

        if speed > self._deadband:
            if abs(turn) < self._deadband:
                return DriveMode.FORWARD
            elif turn < 0:
                return DriveMode.TURN_LEFT
            else:
                return DriveMode.TURN_RIGHT
        elif speed < -self._deadband:
            if abs(turn) < self._deadband:
                return DriveMode.REVERSE
            elif turn < 0:
                return DriveMode.TURN_LEFT
            else:
                return DriveMode.TURN_RIGHT

        return DriveMode.ARC

    def forward(self, speed: float = 0.5) -> None:
        """Drive straight forward."""
        self.drive(abs(speed), 0.0)

    def reverse(self, speed: float = 0.5) -> None:
        """Drive straight backward."""
        self.drive(-abs(speed), 0.0)

    def turn_left(self, speed: float = 0.4, sharpness: float = 0.5) -> None:
        """Turn left while moving forward."""
        self.drive(abs(speed), -abs(sharpness))

    def turn_right(self, speed: float = 0.4, sharpness: float = 0.5) -> None:
        """Turn right while moving forward."""
        self.drive(abs(speed), abs(sharpness))

    def spin_left(self, speed: float = 0.4) -> None:
        """Spin in place counterclockwise."""
        self.drive(0.0, -abs(speed))

    def spin_right(self, speed: float = 0.4) -> None:
        """Spin in place clockwise."""
        self.drive(0.0, abs(speed))

    def stop(self) -> None:
        """Stop both motors with ramping."""
        with self._lock:
            self._current_speed = 0.0
            self._current_turn = 0.0
            self._mode = DriveMode.STOPPED

        self._left_motor.stop()
        self._right_motor.stop()
        logger.info("Drive: STOP")

    def emergency_stop(self) -> None:
        """Immediate stop — no ramping, instant power cut."""
        with self._lock:
            self._current_speed = 0.0
            self._current_turn = 0.0
            self._mode = DriveMode.STOPPED

        self._left_motor.emergency_stop()
        self._right_motor.emergency_stop()
        logger.warning("Drive: EMERGENCY STOP")

    def set_throttle_limit(self, limit: float) -> None:
        """
        Set maximum throttle percentage. Used by safety watchdog
        to reduce speed during thermal or power warnings.
        limit: 0.0 (no movement) to 1.0 (full speed available)
        """
        limit = max(0.0, min(1.0, limit))
        with self._lock:
            old: float = self._throttle_limit
            self._throttle_limit = limit

        if limit != old:
            logger.info("Throttle limit changed: %.0f%% → %.0f%%", old * 100, limit * 100)

    def set_safe_to_drive(self, safe: bool) -> None:
        """Called by safety watchdog to enable/disable driving."""
        with self._lock:
            self._safe_to_drive = safe

        if not safe:
            self.emergency_stop()
            logger.warning("Drive disabled by safety system")
        else:
            logger.info("Drive re-enabled by safety system")

    def disable(self) -> None:
        """Disable the drive system entirely."""
        self.emergency_stop()
        with self._lock:
            self._enabled = False
        self._left_motor.disable()
        self._right_motor.disable()
        logger.warning("Drive controller DISABLED")

    def enable(self) -> None:
        """Re-enable the drive system."""
        with self._lock:
            self._enabled = True
        self._left_motor.enable()
        self._right_motor.enable()
        logger.info("Drive controller enabled")

    def notify_motors_active(self, safety_watchdog: Any) -> None:
        """Report motor state to safety watchdog for runtime tracking."""
        self._safety_callback = safety_watchdog
        is_moving: bool = self._left_motor.is_moving or self._right_motor.is_moving
        safety_watchdog.notify_motors_active(is_moving)

    @property
    def state(self) -> DriveState:
        """Get current drive state snapshot."""
        with self._lock:
            return DriveState(
                mode=self._mode,
                speed=self._current_speed,
                turn=self._current_turn,
                left_speed=self._left_motor.current_speed,
                right_speed=self._right_motor.current_speed,
                throttle_limit=self._throttle_limit,
                enabled=self._enabled,
                safe_to_drive=self._safe_to_drive
            )

    @property
    def left_motor(self) -> MotorDriver:
        return self._left_motor

    @property
    def right_motor(self) -> MotorDriver:
        return self._right_motor

    @property
    def is_moving(self) -> bool:
        return self._left_motor.is_moving or self._right_motor.is_moving

    @property
    def mode(self) -> DriveMode:
        with self._lock:
            return self._mode

    def release(self) -> None:
        """Release all motor resources. Called during shutdown."""
        self.emergency_stop()
        self._left_motor.release()
        self._right_motor.release()
        logger.info("Drive controller resources released")

    def __repr__(self) -> str:
        return (
            f"DriveController(mode={self._mode.value}, "
            f"L={self._left_motor.current_speed:.2f}, "
            f"R={self._right_motor.current_speed:.2f}, "
            f"throttle={self._throttle_limit:.0%})"
        )