"""
Cerberus Brain — Main Orchestrator
The central nervous system. Boots all subsystems in the correct order,
wires them together, runs the mission loop, and manages graceful shutdown.
This is the entry point for Cerberus. Everything starts and ends here.
"""

import os
import sys
import signal
import time
import logging
import threading
from typing import Any, Optional

from cerberus.core.config import CerberusConfig, ConfigError
from cerberus.core.logger import setup_logging, get_logger, get_mqtt_log_handler
from cerberus.core.health import HealthMonitor
from cerberus.core.safety import SafetyWatchdog, SafetyAction, SafetyViolation, SafetyState
from cerberus.storage.db import CerberusDB
from cerberus.comms.mqtt_client import CerberusMQTT


logger: logging.Logger = logging.getLogger(__name__)


class BrainState:
    """Tracks the current operational state of Cerberus."""
    BOOTING: str = "booting"
    INITIALIZING: str = "initializing"
    READY: str = "ready"
    MISSION_ACTIVE: str = "mission_active"
    RTB: str = "rtb"
    SAFE_MODE: str = "safe_mode"
    SHUTTING_DOWN: str = "shutting_down"
    OFFLINE: str = "offline"


class CerberusBrain:
    """
    Main orchestrator for the Cerberus Autonomous AI Rover.
    Manages the full lifecycle: boot → init → ready → mission → shutdown.
    Singleton — there is only one brain.
    """

    _instance: Optional["CerberusBrain"] = None

    def __new__(cls, config_path: Optional[str] = None) -> "CerberusBrain":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._state: str = BrainState.BOOTING
        self._config_path: Optional[str] = config_path
        self._config: Optional[CerberusConfig] = None
        self._db: Optional[CerberusDB] = None
        self._health: Optional[HealthMonitor] = None
        self._safety: Optional[SafetyWatchdog] = None
        self._mqtt: Optional[CerberusMQTT] = None

        self._running: bool = False
        self._shutdown_event: threading.Event = threading.Event()
        self._main_loop_interval: float = 1.0
        self._boot_time: float = time.time()

        self._initialized: bool = True

    def boot(self) -> bool:
        """
        Boot sequence — initialize all subsystems in dependency order.
        Returns True if boot was successful, False on critical failure.
        """
        self._state = BrainState.INITIALIZING
        logger.info("=" * 60)
        logger.info("CERBERUS AUTONOMOUS AI ROVER — BOOTING")
        logger.info("=" * 60)

        try:
            self._init_config()
            self._init_logging()
            self._init_database()
            self._init_health()
            self._init_mqtt()
            self._init_safety()
            self._wire_subsystems()
            self._register_signals()
            self._log_boot_complete()

            self._state = BrainState.READY
            return True

        except ConfigError as e:
            logger.critical("Configuration error during boot: %s", e)
            self._state = BrainState.OFFLINE
            return False
        except Exception as e:
            logger.critical("Fatal error during boot: %s", e, exc_info=True)
            self._state = BrainState.OFFLINE
            return False

    def _init_config(self) -> None:
        """Load and validate master configuration."""
        logger.info("[BOOT] Loading configuration...")
        self._config = CerberusConfig(self._config_path)
        logger.info("[BOOT] Configuration loaded: %s v%s",
                     self._config.get("system", "name"),
                     self._config.get("system", "version"))

    def _init_logging(self) -> None:
        """Initialize the centralized logging system."""
        logger.info("[BOOT] Initializing logging...")
        setup_logging(self._config)
        logger.info("[BOOT] Logging initialized")

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        logger.info("[BOOT] Initializing database...")
        self._db = CerberusDB(self._config)
        self._db.log_system_event(
            event_type="boot",
            source="brain",
            message="Cerberus database initialized",
            severity="INFO"
        )
        logger.info("[BOOT] Database ready")

    def _init_health(self) -> None:
        """Initialize system health monitor."""
        logger.info("[BOOT] Initializing health monitor...")
        self._health = HealthMonitor(self._config)
        self._health.bind_db(self._db)
        self._health.start()
        logger.info("[BOOT] Health monitor active")

    def _init_mqtt(self) -> None:
        """Initialize MQTT communications. Non-blocking — rover works without it."""
        logger.info("[BOOT] Initializing MQTT...")
        self._mqtt = CerberusMQTT(self._config)
        connected: bool = self._mqtt.connect()

        if connected:
            if self._mqtt.wait_for_connection(timeout=5.0):
                logger.info("[BOOT] MQTT connected")
            else:
                logger.warning("[BOOT] MQTT connection pending — will retry in background")
        else:
            logger.warning("[BOOT] MQTT unavailable — operating offline")

    def _init_safety(self) -> None:
        """Initialize safety watchdog. This cannot fail — it's non-negotiable."""
        logger.info("[BOOT] Initializing safety watchdog...")
        self._safety = SafetyWatchdog(self._config, self._health)
        self._safety.bind_db(self._db)
        self._safety.bind_mqtt(self._mqtt)
        self._safety.start()
        logger.info("[BOOT] Safety watchdog ACTIVE")

    def _wire_subsystems(self) -> None:
        """Connect subsystems that depend on each other."""
        logger.info("[BOOT] Wiring subsystems...")

        mqtt_log_handler = get_mqtt_log_handler()
        if mqtt_log_handler is not None and self._mqtt is not None:
            mqtt_log_handler.bind_mqtt(self._mqtt)

        self._health.register_callback(self._on_health_snapshot)

        self._safety.register_action(
            SafetyAction.SAFE_SHUTDOWN,
            self._on_safety_shutdown
        )
        self._safety.register_action(
            SafetyAction.EMERGENCY_STOP,
            self._on_emergency_stop
        )

        if self._mqtt is not None:
            self._mqtt.register_command_handler(
                "cerberus/command/shutdown",
                self._on_command_shutdown
            )
            self._mqtt.register_command_handler(
                "cerberus/command/status",
                self._on_command_status
            )

        logger.info("[BOOT] Subsystems wired")

    def _register_signals(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("[BOOT] Signal handlers registered (SIGINT, SIGTERM)")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle OS signals for graceful shutdown."""
        sig_name: str = signal.Signals(signum).name
        logger.info("Received signal %s — initiating shutdown", sig_name)
        self.shutdown()

    def _log_boot_complete(self) -> None:
        """Log successful boot with timing."""
        elapsed: float = time.time() - self._boot_time
        boot_msg: str = f"Cerberus boot complete in {elapsed:.2f}s"
        logger.info("[BOOT] %s", boot_msg)
        logger.info("=" * 60)

        if self._db is not None:
            self._db.log_system_event(
                event_type="boot_complete",
                source="brain",
                message=boot_msg,
                severity="INFO"
            )

        if self._mqtt is not None:
            self._mqtt.publish_alert({
                "type": "boot_complete",
                "message": boot_msg,
                "state": self._state,
                "boot_time_seconds": round(elapsed, 2)
            })

    def _on_health_snapshot(self, snapshot: Any) -> None:
        """Callback: publish health telemetry to MQTT on each snapshot."""
        if self._mqtt is not None and self._mqtt.is_connected:
            try:
                self._mqtt.publish_health(snapshot.to_dict())
            except Exception as e:
                logger.error("Failed to publish health snapshot: %s", e)

    def _on_safety_shutdown(self, violation: SafetyViolation) -> None:
        """Callback: safety watchdog is requesting shutdown."""
        logger.critical("[BRAIN] Safety shutdown requested: %s", violation.message)
        self.shutdown(reason=violation.message)

    def _on_emergency_stop(self, violation: SafetyViolation) -> None:
        """Callback: emergency stop — kill motors immediately."""
        logger.critical("[BRAIN] Emergency stop: %s", violation.message)
        self._state = BrainState.SAFE_MODE

        if self._db is not None:
            self._db.log_system_event(
                event_type="emergency_stop",
                source="brain",
                message=violation.message,
                severity="CRITICAL"
            )

    def _on_command_shutdown(self, message: Any) -> None:
        """MQTT command: remote shutdown from Dashboard."""
        logger.info("[BRAIN] Remote shutdown command received")
        self.shutdown(reason="Remote shutdown command from Dashboard")

    def _on_command_status(self, message: Any) -> None:
        """MQTT command: Dashboard requesting status report."""
        if self._mqtt is None:
            return

        status: dict[str, Any] = {
            "state": self._state,
            "uptime_seconds": round(time.time() - self._boot_time, 1),
            "safety_state": self._safety.current_state.value if self._safety else "unknown",
            "mqtt_connected": self._mqtt.is_connected if self._mqtt else False,
            "health_running": self._health.is_running if self._health else False,
            "safety_running": self._safety.is_running if self._safety else False,
        }

        if self._health is not None:
            latest = self._health.latest
            status["battery_pct"] = latest.battery_pct
            status["cpu_temp_c"] = latest.cpu_temp_c

        self._mqtt.publish(
            "cerberus/telemetry/status",
            status
        )

    def run(self) -> None:
        """
        Main event loop. Runs until shutdown is requested.
        Phase 1 keeps this simple — later phases add mission execution.
        """
        if self._state != BrainState.READY:
            logger.error("Cannot run — brain is not in READY state (current: %s)", self._state)
            return

        self._running = True
        logger.info("Cerberus main loop started — state: %s", self._state)

        if self._db is not None:
            self._db.log_system_event(
                event_type="main_loop_start",
                source="brain",
                message="Main event loop started",
                severity="INFO"
            )

        while self._running:
            try:
                if self._safety is not None and self._safety.shutdown_requested:
                    logger.critical("[BRAIN] Safety watchdog requested shutdown")
                    self.shutdown(reason="Safety watchdog shutdown request")
                    break

                self._heartbeat()

            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)

            self._shutdown_event.wait(timeout=self._main_loop_interval)

        logger.info("Cerberus main loop exited")

    def _heartbeat(self) -> None:
        """Periodic tasks that run every main loop iteration."""
        if self._mqtt is not None and self._mqtt.is_connected:
            self._mqtt.publish(
                "cerberus/telemetry/heartbeat",
                {
                    "state": self._state,
                    "uptime": round(time.time() - self._boot_time, 1),
                    "safety": self._safety.current_state.value if self._safety else "unknown"
                }
            )

    def shutdown(self, reason: str = "Shutdown requested") -> None:
        """
        Graceful shutdown sequence. Stops all subsystems in reverse
        dependency order and releases all resources.
        """
        if self._state == BrainState.SHUTTING_DOWN:
            return

        self._state = BrainState.SHUTTING_DOWN
        self._running = False
        self._shutdown_event.set()

        logger.info("=" * 60)
        logger.info("CERBERUS SHUTDOWN — Reason: %s", reason)
        logger.info("=" * 60)

        if self._db is not None:
            try:
                self._db.log_system_event(
                    event_type="shutdown",
                    source="brain",
                    message=f"Shutdown: {reason}",
                    severity="INFO"
                )
            except Exception:
                pass

        if self._mqtt is not None:
            try:
                self._mqtt.publish_alert({
                    "type": "shutdown",
                    "message": reason,
                    "uptime_seconds": round(time.time() - self._boot_time, 1)
                })
                time.sleep(0.5)
            except Exception:
                pass

        if self._safety is not None:
            logger.info("[SHUTDOWN] Stopping safety watchdog...")
            try:
                self._safety.stop()
            except Exception as e:
                logger.error("Error stopping safety watchdog: %s", e)

        if self._health is not None:
            logger.info("[SHUTDOWN] Stopping health monitor...")
            try:
                self._health.stop()
            except Exception as e:
                logger.error("Error stopping health monitor: %s", e)

        if self._mqtt is not None:
            logger.info("[SHUTDOWN] Disconnecting MQTT...")
            try:
                self._mqtt.disconnect()
            except Exception as e:
                logger.error("Error disconnecting MQTT: %s", e)

        if self._db is not None:
            logger.info("[SHUTDOWN] Closing database...")
            try:
                self._db.close_all()
            except Exception as e:
                logger.error("Error closing database: %s", e)

        elapsed: float = time.time() - self._boot_time
        logger.info("Cerberus shutdown complete — total uptime: %.1fs", elapsed)
        logger.info("=" * 60)

        self._state = BrainState.OFFLINE

    @property
    def state(self) -> str:
        return self._state

    @property
    def config(self) -> Optional[CerberusConfig]:
        return self._config

    @property
    def db(self) -> Optional[CerberusDB]:
        return self._db

    @property
    def health(self) -> Optional[HealthMonitor]:
        return self._health

    @property
    def safety(self) -> Optional[SafetyWatchdog]:
        return self._safety

    @property
    def mqtt(self) -> Optional[CerberusMQTT]:
        return self._mqtt

    @property
    def uptime(self) -> float:
        return time.time() - self._boot_time

    @classmethod
    def reset(cls) -> None:
        """Reset singleton for testing. Not for production use."""
        cls._instance = None

    def __repr__(self) -> str:
        return (
            f"CerberusBrain(state='{self._state}', "
            f"uptime={self.uptime:.1f}s)"
        )


def main(config_path: Optional[str] = None) -> None:
    """Entry point for Cerberus."""
    brain: CerberusBrain = CerberusBrain(config_path)

    if not brain.boot():
        logger.critical("Boot failed — Cerberus cannot start")
        sys.exit(1)

    try:
        brain.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        brain.shutdown(reason="Process exit")
        sys.exit(0)


if __name__ == "__main__":
    config: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
    main(config)