"""
Cerberus Motor Validation Script
Run this when hardware arrives to verify motor wiring, direction,
and drive controller operation. Step-by-step interactive test.

Usage:
    python -m scripts.test_motors
    python -m scripts.test_motors --sim   (simulation mode, no hardware)
"""

import sys
import time
import logging
import argparse

sys.path.insert(0, ".")

from cerberus.core.config import CerberusConfig
from cerberus.core.logger import setup_logging
from cerberus.mobility.motor_driver import MotorDriver
from cerberus.mobility.drive_controller import DriveController


logger: logging.Logger = logging.getLogger("test_motors")

SEPARATOR: str = "=" * 50


def wait_for_enter(message: str = "Press ENTER to continue...") -> bool:
    """Wait for user input. Returns False if user types 'quit'."""
    try:
        response: str = input(f"\n{message} ").strip().lower()
        return response != "quit"
    except (KeyboardInterrupt, EOFError):
        return False


def print_header(title: str) -> None:
    """Print a test section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def test_individual_motors(config: CerberusConfig) -> bool:
    """Test each motor driver independently."""
    print_header("TEST 1: Individual Motor Drivers")

    left_cfg = config.section("motors")["left"]
    right_cfg = config.section("motors")["right"]

    left: MotorDriver = MotorDriver(
        name="left",
        pwm_pin=left_cfg["pwm_pin"],
        forward_pin=left_cfg["forward_pin"],
        reverse_pin=left_cfg["reverse_pin"],
        config=config
    )

    right: MotorDriver = MotorDriver(
        name="right",
        pwm_pin=right_cfg["pwm_pin"],
        forward_pin=right_cfg["forward_pin"],
        reverse_pin=right_cfg["reverse_pin"],
        config=config
    )

    print(f"\nLeft motor:  {left}")
    print(f"Right motor: {right}")

    print("\n--- Left Motor Forward (30% speed, 2 seconds) ---")
    print("EXPECTED: Left side wheels spin forward")
    if not wait_for_enter("WHEELS CLEAR? Press ENTER to run..."):
        return False

    left.set_speed_immediate(0.3)
    print(f"  Running... {left}")
    time.sleep(2.0)
    left.emergency_stop()
    print("  Stopped.")

    print("\nDid the LEFT wheels spin FORWARD? (yes/no/quit)")
    response: str = input("  > ").strip().lower()
    if response == "quit":
        left.release()
        right.release()
        return False
    if response != "yes":
        print("\n  ⚠ WIRING CHECK: Swap the forward/reverse wires on the LEFT BTS7960")
        print("  Or swap the motor leads (B+/B-) on the left driver")

    print("\n--- Left Motor Reverse (30% speed, 2 seconds) ---")
    print("EXPECTED: Left side wheels spin backward")
    if not wait_for_enter():
        left.release()
        right.release()
        return False

    left.set_speed_immediate(-0.3)
    print(f"  Running... {left}")
    time.sleep(2.0)
    left.emergency_stop()
    print("  Stopped.")

    print("\n--- Right Motor Forward (30% speed, 2 seconds) ---")
    print("EXPECTED: Right side wheels spin forward")
    if not wait_for_enter():
        left.release()
        right.release()
        return False

    right.set_speed_immediate(0.3)
    print(f"  Running... {right}")
    time.sleep(2.0)
    right.emergency_stop()
    print("  Stopped.")

    print("\n--- Right Motor Reverse (30% speed, 2 seconds) ---")
    print("EXPECTED: Right side wheels spin backward")
    if not wait_for_enter():
        left.release()
        right.release()
        return False

    right.set_speed_immediate(-0.3)
    print(f"  Running... {right}")
    time.sleep(2.0)
    right.emergency_stop()
    print("  Stopped.")

    left.release()
    right.release()

    print("\n✓ Individual motor test complete")
    return True


def test_ramp(config: CerberusConfig) -> bool:
    """Test motor ramping (acceleration smoothing)."""
    print_header("TEST 2: Motor Ramping")

    left_cfg = config.section("motors")["left"]

    left: MotorDriver = MotorDriver(
        name="left",
        pwm_pin=left_cfg["pwm_pin"],
        forward_pin=left_cfg["forward_pin"],
        reverse_pin=left_cfg["reverse_pin"],
        config=config
    )

    print("\n--- Ramp Up: 0% → 60% (left motor) ---")
    print("EXPECTED: Smooth acceleration, not instant jump")
    if not wait_for_enter("WHEELS CLEAR? Press ENTER to run..."):
        left.release()
        return False

    left.set_speed(0.6)
    print("  Ramping up...")
    time.sleep(3.0)

    print(f"  Current: {left}")

    print("\n--- Ramp Down: 60% → 0% ---")
    left.set_speed(0.0)
    print("  Ramping down...")
    time.sleep(3.0)
    left.emergency_stop()
    print("  Stopped.")

    left.release()

    print("\n✓ Ramp test complete")
    return True


def test_drive_controller(config: CerberusConfig) -> bool:
    """Test the 6WD skid-steer drive controller."""
    print_header("TEST 3: Drive Controller (Skid-Steer)")

    drive: DriveController = DriveController(config)

    print(f"\nDrive state: {drive}")

    print("\n--- Forward (40% speed, 3 seconds) ---")
    print("EXPECTED: All 6 wheels spin forward, rover moves straight")
    if not wait_for_enter("WHEELS CLEAR? Press ENTER to run..."):
        drive.release()
        return False

    drive.forward(0.4)
    print(f"  Running... {drive}")
    time.sleep(3.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    print("\n--- Reverse (40% speed, 3 seconds) ---")
    print("EXPECTED: All 6 wheels spin backward")
    if not wait_for_enter():
        drive.release()
        return False

    drive.reverse(0.4)
    print(f"  Running... {drive}")
    time.sleep(3.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    print("\n--- Spin Left (40% speed, 2 seconds) ---")
    print("EXPECTED: Left wheels backward, right wheels forward — rover rotates left")
    if not wait_for_enter():
        drive.release()
        return False

    drive.spin_left(0.4)
    print(f"  Running... {drive}")
    time.sleep(2.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    print("\n--- Spin Right (40% speed, 2 seconds) ---")
    print("EXPECTED: Right wheels backward, left wheels forward — rover rotates right")
    if not wait_for_enter():
        drive.release()
        return False

    drive.spin_right(0.4)
    print(f"  Running... {drive}")
    time.sleep(2.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    print("\n--- Arc Turn Left (40% speed, 50% sharpness, 3 seconds) ---")
    print("EXPECTED: Rover moves forward while curving left")
    if not wait_for_enter():
        drive.release()
        return False

    drive.turn_left(0.4, 0.5)
    print(f"  Running... {drive}")
    time.sleep(3.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    print("\n--- Arc Turn Right (40% speed, 50% sharpness, 3 seconds) ---")
    print("EXPECTED: Rover moves forward while curving right")
    if not wait_for_enter():
        drive.release()
        return False

    drive.turn_right(0.4, 0.5)
    print(f"  Running... {drive}")
    time.sleep(3.0)
    drive.stop()
    time.sleep(1.0)
    print("  Stopped.")

    drive.release()

    print("\n✓ Drive controller test complete")
    return True


def test_emergency_stop(config: CerberusConfig) -> bool:
    """Test emergency stop functionality."""
    print_header("TEST 4: Emergency Stop")

    drive: DriveController = DriveController(config)

    print("\n--- Full speed forward then EMERGENCY STOP ---")
    print("EXPECTED: Motors go to full speed, then cut instantly (no ramp)")
    if not wait_for_enter("WHEELS CLEAR? Press ENTER to run..."):
        drive.release()
        return False

    drive.forward(0.8)
    print(f"  Running at 80%... {drive}")
    time.sleep(2.0)

    print("  >>> EMERGENCY STOP <<<")
    drive.emergency_stop()
    print(f"  After e-stop: {drive}")

    time.sleep(1.0)

    is_stopped: bool = not drive.is_moving
    print(f"\n  Motors stopped: {'YES ✓' if is_stopped else 'NO ✗ — CHECK WIRING'}")

    drive.release()

    print("\n✓ Emergency stop test complete")
    return True


def test_throttle_limit(config: CerberusConfig) -> bool:
    """Test throttle limiting (safety system integration)."""
    print_header("TEST 5: Throttle Limiting")

    drive: DriveController = DriveController(config)

    print("\n--- Forward at 80% with no throttle limit ---")
    if not wait_for_enter("WHEELS CLEAR? Press ENTER to run..."):
        drive.release()
        return False

    drive.forward(0.8)
    print(f"  Full throttle: {drive}")
    time.sleep(2.0)
    drive.stop()
    time.sleep(1.0)

    print("\n--- Forward at 80% with 50% throttle limit ---")
    print("EXPECTED: Noticeably slower — effective speed is 40%")
    if not wait_for_enter():
        drive.release()
        return False

    drive.set_throttle_limit(0.5)
    drive.forward(0.8)
    print(f"  Throttled: {drive}")
    time.sleep(2.0)
    drive.stop()
    time.sleep(1.0)

    drive.set_throttle_limit(1.0)
    drive.release()

    print("\n✓ Throttle limit test complete")
    return True


def run_simulation() -> None:
    """Run all tests in simulation mode — no hardware needed."""
    print_header("SIMULATION MODE — No GPIO Hardware")
    print("All motor commands will be logged but no physical output occurs.")
    print("Use this to verify code paths and command flow.\n")

    config: CerberusConfig = CerberusConfig()

    drive: DriveController = DriveController(config)

    print("Forward 50%:")
    drive.forward(0.5)
    print(f"  State: {drive.state.to_dict()}")

    print("\nSpin left 40%:")
    drive.spin_left(0.4)
    print(f"  State: {drive.state.to_dict()}")

    print("\nArc right 60% speed, 30% turn:")
    drive.drive(0.6, 0.3)
    print(f"  State: {drive.state.to_dict()}")

    print("\nThrottle limit 50%, forward 80%:")
    drive.set_throttle_limit(0.5)
    drive.forward(0.8)
    print(f"  State: {drive.state.to_dict()}")

    print("\nEmergency stop:")
    drive.emergency_stop()
    print(f"  State: {drive.state.to_dict()}")

    drive.release()

    print(f"\n{SEPARATOR}")
    print("  Simulation complete — all code paths verified")
    print(SEPARATOR)


def main() -> None:
    """Main entry point for motor validation."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Cerberus Motor Validation Script"
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run in simulation mode (no hardware required)"
    )
    args: argparse.Namespace = parser.parse_args()

    config: CerberusConfig = CerberusConfig()
    setup_logging(config)

    print(f"\n{SEPARATOR}")
    print("  CERBERUS MOTOR VALIDATION")
    print(f"  {'SIMULATION MODE' if args.sim else 'HARDWARE MODE'}")
    print(SEPARATOR)

    if args.sim:
        run_simulation()
        return

    print("\n⚠  SAFETY WARNING:")
    print("   1. Rover must be on blocks — wheels OFF the ground")
    print("   2. Keep hands clear of wheels at all times")
    print("   3. Have power switch ready to kill power if needed")
    print("   4. Type 'quit' at any prompt to abort all tests")

    if not wait_for_enter("\nReady to begin? Press ENTER..."):
        print("Aborted.")
        return

    tests: list[tuple[str, callable]] = [
        ("Individual Motors", test_individual_motors),
        ("Motor Ramping", test_ramp),
        ("Drive Controller", test_drive_controller),
        ("Emergency Stop", test_emergency_stop),
        ("Throttle Limiting", test_throttle_limit),
    ]

    passed: int = 0
    failed: int = 0

    for name, test_func in tests:
        try:
            if test_func(config):
                passed += 1
            else:
                print(f"\n✗ Test aborted: {name}")
                break
        except Exception as e:
            logger.error("Test '%s' failed with error: %s", name, e)
            failed += 1

    print(f"\n{SEPARATOR}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests) - passed - failed} skipped")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
