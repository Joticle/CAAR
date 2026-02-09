"""
Cerberus System Health Monitor
Continuously monitors CPU temperature, CPU usage, memory, disk,
and power system (INA3221). Publishes snapshots to SQLite and MQTT.
Runs as a background thread — always watching, always logging.
"""

import time
import threading
import logging
from typing import Any, Optional
from dataclasses import dataclass, asdict

import psutil

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Immutable snapshot of system health at a point in time."""
    cpu_temp_c: float = 0.0
    cpu_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    disk_usage_pct: float = 0.0
    battery_voltage: float = 0.0
    battery_current_a: float = 0.0
    battery_pct: float = 100.0
    pi_rail_voltage: float = 5.0
    motor_current_a: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_fix: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PowerMonitor:
    """
    Reads voltage, current, and power from INA3221 three-channel monitor.
    Gracefully degrades if hardware is not present (dev environment).
    """

    def __init__(self, config: CerberusConfig) -> None:
        self._enabled: bool = False
        self._full_voltage: float = config.get("power", "channels", "battery", "full_voltage", default=12.6)
        self._empty_voltage: float = config.get("power", "channels", "battery", "empty_voltage", default=9.0)
        self._ina: Optional[Any] = None
        self._init_hardware()

    def _init_hardware(self) -> None:
        """Attempt to initialize INA3221. Fails gracefully on dev machines."""
        try:
            from SDL_Pi_INA3221 import SDL_Pi_INA3221
            self._ina = SDL_Pi_INA3221(addr=0x40)
            self._enabled = True
            logger.info("INA3221 power monitor initialized")
        except ImportError:
            logger.warning("INA3221 library not available — power monitoring disabled (dev mode)")
        except Exception as e:
            logger.error("INA3221 initialization failed: %s", e)

    def read_battery_voltage(self) -> float:
        """Read battery pack voltage from channel 1."""
        if not self._enabled or self._ina is None:
            return 0.0
        try:
            return float(self._ina.getBusVoltage_V(1))
        except Exception as e:
            logger.error("Failed to read battery voltage: %s", e)
            return 0.0

    def read_battery_current(self) -> float:
        """Read battery current draw from channel 1 in amps."""
        if not self._enabled or self._ina is None:
            return 0.0
        try:
            return float(self._ina.getCurrent_mA(1)) / 1000.0
        except Exception as e:
            logger.error("Failed to read battery current: %s", e)
            return 0.0

    def read_pi_rail_voltage(self) -> float:
        """Read Pi 5V rail voltage from channel 2."""
        if not self._enabled or self._ina is None:
            return 0.0
        try:
            return float(self._ina.getBusVoltage_V(2))
        except Exception as e:
            logger.error("Failed to read Pi rail voltage: %s", e)
            return 0.0

    def read_motor_current(self) -> float:
        """Read motor current draw from channel 3 in amps."""
        if not self._enabled or self._ina is None:
            return 0.0
        try:
            return float(self._ina.getCurrent_mA(3)) / 1000.0
        except Exception as e:
            logger.error("Failed to read motor current: %s", e)
            return 0.0

    def estimate_battery_pct(self, voltage: float) -> float:
        """Estimate battery percentage from voltage using linear interpolation."""
        if voltage <= 0.0:
            return 0.0
        if voltage >= self._full_voltage:
            return 100.0
        if voltage <= self._empty_voltage:
            return 0.0
        range_v: float = self._full_voltage - self._empty_voltage
        current_v: float = voltage - self._empty_voltage
        return round((current_v / range_v) * 100.0, 1)

    @property
    def enabled(self) -> bool:
        return self._enabled


class HealthMonitor:
    """
    Background health monitor for Cerberus.
    Periodically samples system metrics and stores snapshots.
    Other subsystems read the latest snapshot to make decisions.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._interval: int = config.get("health", "report_interval_seconds", default=15)
        self._disk_warn: float = config.get("health", "disk_warning_pct", default=85)
        self._mem_warn: float = config.get("health", "memory_warning_pct", default=85)
        self._cpu_warn: float = config.get("health", "cpu_warning_pct", default=90)

        self._power: PowerMonitor = PowerMonitor(config)
        self._latest: HealthSnapshot = HealthSnapshot()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._db: Optional[Any] = None
        self._callbacks: list = []

        logger.info("Health monitor created — interval=%ds", self._interval)

    def bind_db(self, db: Any) -> None:
        """Bind database for snapshot persistence. Called after DB init."""
        self._db = db

    def register_callback(self, callback) -> None:
        """Register a function to be called with each new snapshot."""
        self._callbacks.append(callback)

    def _read_cpu_temp(self) -> float:
        """Read CPU temperature. Works on Pi, returns 0.0 on dev machines."""
        try:
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps and temps["cpu_thermal"]:
                return float(temps["cpu_thermal"][0].current)
            if "cpu-thermal" in temps and temps["cpu-thermal"]:
                return float(temps["cpu-thermal"][0].current)
            for key in temps:
                if temps[key]:
                    return float(temps[key][0].current)
            return 0.0
        except Exception:
            return 0.0

    def _read_cpu_usage(self) -> float:
        """Read CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    def _read_memory_usage(self) -> float:
        """Read memory usage percentage."""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    def _read_disk_usage(self) -> float:
        """Read disk usage percentage for the root partition."""
        try:
            return psutil.disk_usage("/").percent
        except Exception:
            try:
                return psutil.disk_usage("C:\\").percent
            except Exception:
                return 0.0

    def take_snapshot(self) -> HealthSnapshot:
        """Capture a complete health snapshot right now."""
        battery_v: float = self._power.read_battery_voltage()

        snapshot: HealthSnapshot = HealthSnapshot(
            cpu_temp_c=self._read_cpu_temp(),
            cpu_usage_pct=self._read_cpu_usage(),
            memory_usage_pct=self._read_memory_usage(),
            disk_usage_pct=self._read_disk_usage(),
            battery_voltage=battery_v,
            battery_current_a=self._power.read_battery_current(),
            battery_pct=self._power.estimate_battery_pct(battery_v),
            pi_rail_voltage=self._power.read_pi_rail_voltage(),
            motor_current_a=self._power.read_motor_current(),
        )

        with self._lock:
            self._latest = snapshot

        self._check_warnings(snapshot)
        self._persist(snapshot)
        self._notify(snapshot)

        return snapshot

    def _check_warnings(self, snapshot: HealthSnapshot) -> None:
        """Log warnings when thresholds are exceeded."""
        if snapshot.cpu_usage_pct > self._cpu_warn:
            logger.warning("High CPU usage: %.1f%%", snapshot.cpu_usage_pct)

        if snapshot.memory_usage_pct > self._mem_warn:
            logger.warning("High memory usage: %.1f%%", snapshot.memory_usage_pct)

        if snapshot.disk_usage_pct > self._disk_warn:
            logger.warning("High disk usage: %.1f%%", snapshot.disk_usage_pct)

        if self._power.enabled and snapshot.pi_rail_voltage > 0:
            min_v: float = self._config.get(
                "power", "channels", "pi_rail", "min_voltage", default=4.75
            )
            if snapshot.pi_rail_voltage < min_v:
                logger.warning("Pi 5V rail sagging: %.2fV (min %.2fV)", snapshot.pi_rail_voltage, min_v)

    def _persist(self, snapshot: HealthSnapshot) -> None:
        """Write snapshot to SQLite."""
        if self._db is None:
            return
        try:
            self._db.log_health(snapshot.to_dict())
        except Exception as e:
            logger.error("Failed to persist health snapshot: %s", e)

    def _notify(self, snapshot: HealthSnapshot) -> None:
        """Call all registered callbacks with the new snapshot."""
        for callback in self._callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error("Health callback failed: %s", e)

    @property
    def latest(self) -> HealthSnapshot:
        """Get the most recent health snapshot. Thread-safe."""
        with self._lock:
            return self._latest

    @property
    def power_monitor(self) -> PowerMonitor:
        return self._power

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("Health monitor started")
        psutil.cpu_percent(interval=None)

        while self._running:
            try:
                self.take_snapshot()
            except Exception as e:
                logger.error("Health monitor error: %s", e)

            for _ in range(self._interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("Health monitor stopped")

    def start(self) -> None:
        """Start the background health monitoring thread."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="health-monitor",
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the health monitoring thread gracefully."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            if self._thread.is_alive():
                logger.warning("Health monitor thread did not stop cleanly")
            self._thread = None

        logger.info("Health monitor shutdown complete")

    @property
    def is_running(self) -> bool:
        return self._running

    def __repr__(self) -> str:
        status: str = "running" if self._running else "stopped"
        return f"HealthMonitor(interval={self._interval}s, power={'enabled' if self._power.enabled else 'disabled'}, status={status})"