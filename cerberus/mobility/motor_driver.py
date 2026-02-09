"""
Cerberus BTS7960 Motor Driver
Controls a single BTS7960 dual H-bridge module (one side of the drivetrain).
Handles PWM speed control, direction, ramping, and emergency stop.
Gracefully degrades when GPIO hardware is not present (dev environment).
"""

import time
import threading
import logging
from typing import Any, Optional
from enum import Enum

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class MotorDirection(Enum):
    """Motor rotation direction."""
    STOPPED = "stopped"
    FORWARD = "forward"
    REVERSE = "reverse"


class MotorDriver:
    """
    Controls one BTS7960 dual H-bridge motor driver.
    Each BTS7960 drives one side of the 6WD chassis (3 motors wired in parallel).
    PWM controls speed, two direction pins select forward/reverse.

    BTS7960 wiring:
        RPWM → forward rotation enable
        LPWM → reverse rotation enable
        R_EN + L_EN → tied HIGH (always enabled) or driven by PWM pin
        VCC → 5V logic
        GND → common ground with Pi and battery
        B+ / B- → motor power (from battery through switch)
    """

    def __init__(
        self,
        name: str,
        pwm_pin: int,
        forward_pin: int,
        reverse_pin: int,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._name: str = name
        self._pwm_pin: int = pwm_pin
        self._forward_pin: int = forward_pin
        self._reverse_pin: int = reverse_pin

        self._pwm_frequency: int = config.get("motors", "pwm_frequency", default=1000)
        self._min_duty: float = config.get("motors", "min_duty_cycle", default=0.0)
        self._max_duty: float = config.get("motors", "max_duty_cycle", default=1.0)
        self._ramp_step: float = config.get("motors", "ramp_step", default=0.05)
        self._ramp_interval: float = config.get("motors", "ramp_interval_ms", default=50) / 1000.0
        self._deadband: float = config.get("motors", "deadband", default=0.1)

        self._current_speed: float = 0.0
        self._target_speed: float = 0.0
        self._direction: MotorDirection = MotorDirection.STOPPED
        self._enabled: bool = False
        self._hardware_available: bool = False

        self._lock: threading.Lock = threading.Lock()
        self._ramp_thread: Optional[threading.Thread] = None
        self._ramping: bool = False

        self._pwm_device: Optional[Any] = None
        self._forward_device: Optional[Any] = None
        self._reverse_device: Optional[Any] = None

        self._init_hardware()

        logger.info(
            "Motor driver '%s' created — PWM=GPIO%d, FWD=GPIO%d, REV=GPIO%d, freq=%dHz",
            self._name, self._pwm_pin, self._forward_pin, self._reverse_pin, self._pwm_frequency
        )

    def _init_hardware(self) -> None:
        """Initialize GPIO pins. Fails gracefully on dev machines."""
        try:
            from gpiozero import PWMOutputDevice, DigitalOutputDevice

            self._pwm_device = PWMOutputDevice(
                pin=self._pwm_pin,
                frequency=self._pwm_frequency,
                initial_value=0
            )
            self._forward_device = DigitalOutputDevice(
                pin=self._forward_pin,
                initial_value=False
            )
            self._reverse_device = DigitalOutputDevice(
                pin=self._reverse_pin,
                initial_value=False
            )

            self._hardware_available = True
            self._enabled = True
            logger.info("Motor '%s' GPIO initialized", self._name)

        except ImportError:
            logger.warning(
                "Motor '%s' — gpiozero not available, running in simulation mode",
                self._name
            )
            self._hardware_available = False
            self._enabled = True

        except Exception as e:
            logger.error("Motor '%s' GPIO initialization failed: %s", self._name, e)
            self._hardware_available = False
            self._enabled = False

    def set_speed(self, speed: float) -> None:
        """
        Set motor speed with ramping.
        speed: -1.0 (full reverse) to 1.0 (full forward), 0.0 = stop.
        """
        if not self._enabled:
            logger.warning("Motor '%s' is disabled — ignoring speed command", self._name)
            return

        speed = max(-1.0, min(1.0, speed))

        if abs(speed) < self._deadband:
            speed = 0.0

        with self._lock:
            self._target_speed = speed

        if not self._ramping:
            self._start_ramp()

    def set_speed_immediate(self, speed: float) -> None:
        """Set motor speed immediately without ramping. Use for emergency stops."""
        if not self._enabled:
            return

        speed = max(-1.0, min(1.0, speed))

        if abs(speed) < self._deadband:
            speed = 0.0

        self._stop_ramp()

        with self._lock:
            self._target_speed = speed

        self._apply_speed(speed)

    def _apply_speed(self, speed: float) -> None:
        """Apply speed directly to hardware or simulation."""
        with self._lock:
            if speed > 0:
                direction = MotorDirection.FORWARD
                duty = min(abs(speed), self._max_duty)
            elif speed < 0:
                direction = MotorDirection.REVERSE
                duty = min(abs(speed), self._max_duty)
            else:
                direction = MotorDirection.STOPPED
                duty = 0.0

            self._current_speed = speed
            self._direction = direction

        if self._hardware_available:
            try:
                if direction == MotorDirection.FORWARD:
                    self._reverse_device.off()
                    self._forward_device.on()
                    self._pwm_device.value = duty
                elif direction == MotorDirection.REVERSE:
                    self._forward_device.off()
                    self._reverse_device.on()
                    self._pwm_device.value = duty
                else:
                    self._forward_device.off()
                    self._reverse_device.off()
                    self._pwm_device.value = 0
            except Exception as e:
                logger.error("Motor '%s' hardware error: %s", self._name, e)
                self._emergency_stop_hardware()
        else:
            logger.debug(
                "Motor '%s' [SIM] direction=%s, duty=%.2f",
                self._name, direction.value, duty
            )

    def _start_ramp(self) -> None:
        """Start the ramping thread to smoothly transition speed."""
        if self._ramping:
            return

        self._ramping = True
        self._ramp_thread = threading.Thread(
            target=self._ramp_loop,
            name=f"motor-ramp-{self._name}",
            daemon=True
        )
        self._ramp_thread.start()

    def _stop_ramp(self) -> None:
        """Stop the ramping thread."""
        self._ramping = False
        if self._ramp_thread is not None:
            self._ramp_thread.join(timeout=1.0)
            self._ramp_thread = None

    def _ramp_loop(self) -> None:
        """Smoothly ramp current speed toward target speed."""
        while self._ramping:
            with self._lock:
                current: float = self._current_speed
                target: float = self._target_speed

            if abs(current - target) < self._ramp_step:
                self._apply_speed(target)
                self._ramping = False
                break

            if target > current:
                new_speed: float = current + self._ramp_step
            else:
                new_speed = current - self._ramp_step

            self._apply_speed(new_speed)
            time.sleep(self._ramp_interval)

    def stop(self) -> None:
        """Stop the motor with ramping."""
        self.set_speed(0.0)

    def emergency_stop(self) -> None:
        """Immediate stop — no ramping, kills power to motor instantly."""
        self._stop_ramp()

        with self._lock:
            self._current_speed = 0.0
            self._target_speed = 0.0
            self._direction = MotorDirection.STOPPED

        if self._hardware_available:
            self._emergency_stop_hardware()

        logger.warning("Motor '%s' EMERGENCY STOP", self._name)

    def _emergency_stop_hardware(self) -> None:
        """Kill all GPIO outputs immediately."""
        try:
            if self._pwm_device is not None:
                self._pwm_device.value = 0
            if self._forward_device is not None:
                self._forward_device.off()
            if self._reverse_device is not None:
                self._reverse_device.off()
        except Exception as e:
            logger.error("Motor '%s' emergency stop hardware error: %s", self._name, e)

    def disable(self) -> None:
        """Disable the motor driver. Emergency stop then prevent further commands."""
        self.emergency_stop()
        self._enabled = False
        logger.warning("Motor '%s' DISABLED", self._name)

    def enable(self) -> None:
        """Re-enable the motor driver after being disabled."""
        self._enabled = True
        logger.info("Motor '%s' enabled", self._name)

    def release(self) -> None:
        """Release all GPIO resources. Called during shutdown."""
        self._stop_ramp()
        self.emergency_stop()

        if self._hardware_available:
            try:
                if self._pwm_device is not None:
                    self._pwm_device.close()
                if self._forward_device is not None:
                    self._forward_device.close()
                if self._reverse_device is not None:
                    self._reverse_device.close()
                logger.info("Motor '%s' GPIO resources released", self._name)
            except Exception as e:
                logger.error("Motor '%s' GPIO release error: %s", self._name, e)

        self._enabled = False

    @property
    def current_speed(self) -> float:
        with self._lock:
            return self._current_speed

    @property
    def target_speed(self) -> float:
        with self._lock:
            return self._target_speed

    @property
    def direction(self) -> MotorDirection:
        with self._lock:
            return self._direction

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._direction != MotorDirection.STOPPED

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def hardware_available(self) -> bool:
        return self._hardware_available

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return (
            f"MotorDriver(name='{self._name}', speed={self._current_speed:.2f}, "
            f"direction={self._direction.value}, "
            f"hw={'yes' if self._hardware_available else 'sim'})"
        )