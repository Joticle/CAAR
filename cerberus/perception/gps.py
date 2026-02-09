"""
Cerberus GPS Interface
Reads position data from a u-blox NEO-6M GPS module via gpsd.
Provides lat/lon, altitude, speed, heading, and fix quality.
Feeds the navigator and geotags detections.
Gracefully degrades when GPS hardware is not present (dev environment).
"""

import time
import logging
import threading
from typing import Any, Optional
from dataclasses import dataclass

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class GPSFixType:
    """GPS fix quality levels."""
    NO_FIX: int = 0
    FIX_2D: int = 2
    FIX_3D: int = 3


@dataclass
class GPSReading:
    """Snapshot of current GPS data."""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    fix_type: int = GPSFixType.NO_FIX
    satellites: int = 0
    hdop: float = 99.9
    timestamp: float = 0.0

    @property
    def has_fix(self) -> bool:
        return self.fix_type >= GPSFixType.FIX_2D

    @property
    def has_3d_fix(self) -> bool:
        return self.fix_type >= GPSFixType.FIX_3D

    @property
    def accuracy_good(self) -> bool:
        return self.has_fix and self.hdop < 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "alt_m": round(self.alt_m, 1),
            "speed_mps": round(self.speed_mps, 2),
            "heading_deg": round(self.heading_deg, 1),
            "fix_type": self.fix_type,
            "satellites": self.satellites,
            "hdop": round(self.hdop, 1),
            "has_fix": self.has_fix,
            "timestamp": self.timestamp
        }


class CerberusGPS:
    """
    GPS interface for Cerberus using gpsd.
    Runs a background polling thread that continuously reads position.
    Other subsystems read the latest position via the reading property.

    gpsd must be configured and running on the Pi:
        sudo gpsd /dev/ttyAMA0 -F /var/run/gpsd.sock

    On dev machines, simulates a fixed position (Las Vegas).
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._poll_interval: float = config.get("navigation", "gps_poll_interval", default=1.0)
        self._gps_timeout: int = config.get("navigation", "gps_timeout_seconds", default=10)
        self._sim_lat: float = config.get("navigation", "home_lat", default=36.1699)
        self._sim_lon: float = config.get("navigation", "home_lon", default=-115.1398)

        self._gpsd_session: Optional[Any] = None
        self._hardware_available: bool = False
        self._latest: GPSReading = GPSReading()
        self._lock: threading.Lock = threading.Lock()

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._position_callbacks: list = []
        self._fix_acquired_callbacks: list = []
        self._fix_lost_callbacks: list = []
        self._had_fix: bool = False

        self._init_hardware()

        logger.info(
            "GPS created — poll_interval=%.1fs, hw=%s",
            self._poll_interval,
            "yes" if self._hardware_available else "sim"
        )

    def _init_hardware(self) -> None:
        """Initialize gpsd connection. Fails gracefully on dev machines."""
        try:
            from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE

            self._gpsd_session = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)
            self._hardware_available = True
            logger.info("gpsd connection established")

        except ImportError:
            logger.warning("gpsd library not available — GPS running in simulation mode")
            self._hardware_available = False

        except Exception as e:
            logger.error("gpsd connection failed: %s", e)
            self._hardware_available = False

    def start(self) -> None:
        """Start the GPS polling thread."""
        if self._running:
            logger.warning("GPS already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gps-poller",
            daemon=True
        )
        self._thread.start()
        logger.info("GPS polling started")

    def stop(self) -> None:
        """Stop the GPS polling thread."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 2)
            if self._thread.is_alive():
                logger.warning("GPS thread did not stop cleanly")
            self._thread = None

        logger.info("GPS polling stopped")

    def _poll_loop(self) -> None:
        """Background loop that reads GPS data from gpsd or simulation."""
        logger.info("GPS poll loop started")

        while self._running:
            try:
                if self._hardware_available:
                    self._read_gpsd()
                else:
                    self._read_simulated()

            except Exception as e:
                logger.error("GPS poll error: %s", e)

            time.sleep(self._poll_interval)

        logger.info("GPS poll loop stopped")

    def _read_gpsd(self) -> None:
        """Read position data from gpsd."""
        try:
            report = self._gpsd_session.next()

            if report.get("class") != "TPV":
                return

            fix_mode: int = report.get("mode", 1)
            if fix_mode < 2:
                fix_type: int = GPSFixType.NO_FIX
            elif fix_mode == 2:
                fix_type = GPSFixType.FIX_2D
            else:
                fix_type = GPSFixType.FIX_3D

            lat: float = report.get("lat", 0.0)
            lon: float = report.get("lon", 0.0)

            if lat == 0.0 and lon == 0.0:
                fix_type = GPSFixType.NO_FIX

            reading: GPSReading = GPSReading(
                lat=lat,
                lon=lon,
                alt_m=report.get("alt", 0.0),
                speed_mps=report.get("speed", 0.0),
                heading_deg=report.get("track", 0.0),
                fix_type=fix_type,
                satellites=self._get_satellite_count(),
                hdop=self._get_hdop(),
                timestamp=time.time()
            )

            self._update_reading(reading)

        except StopIteration:
            pass
        except Exception as e:
            logger.error("gpsd read error: %s", e)

    def _get_satellite_count(self) -> int:
        """Get the number of satellites in view from gpsd."""
        try:
            if hasattr(self._gpsd_session, "satellites"):
                return len([s for s in self._gpsd_session.satellites if s.used])
            return 0
        except Exception:
            return 0

    def _get_hdop(self) -> float:
        """Get horizontal dilution of precision from gpsd."""
        try:
            if hasattr(self._gpsd_session, "hdop"):
                hdop = self._gpsd_session.hdop
                if isinstance(hdop, (int, float)) and hdop > 0:
                    return float(hdop)
            return 99.9
        except Exception:
            return 99.9

    def _read_simulated(self) -> None:
        """Generate simulated GPS readings for dev environment."""
        reading: GPSReading = GPSReading(
            lat=self._sim_lat,
            lon=self._sim_lon,
            alt_m=620.0,
            speed_mps=0.0,
            heading_deg=0.0,
            fix_type=GPSFixType.FIX_3D,
            satellites=8,
            hdop=1.2,
            timestamp=time.time()
        )

        self._update_reading(reading)

    def _update_reading(self, reading: GPSReading) -> None:
        """Update the latest reading and fire callbacks."""
        with self._lock:
            self._latest = reading

        if reading.has_fix and not self._had_fix:
            self._had_fix = True
            logger.info(
                "GPS fix acquired: %.7f, %.7f (%d satellites, HDOP=%.1f)",
                reading.lat, reading.lon, reading.satellites, reading.hdop
            )
            for cb in self._fix_acquired_callbacks:
                try:
                    cb(reading)
                except Exception as e:
                    logger.error("GPS fix acquired callback error: %s", e)

        elif not reading.has_fix and self._had_fix:
            self._had_fix = False
            logger.warning("GPS fix lost")
            for cb in self._fix_lost_callbacks:
                try:
                    cb()
                except Exception as e:
                    logger.error("GPS fix lost callback error: %s", e)

        if reading.has_fix:
            for cb in self._position_callbacks:
                try:
                    cb(reading.lat, reading.lon)
                except Exception as e:
                    logger.error("GPS position callback error: %s", e)

    def register_position_callback(self, callback) -> None:
        """Register callback for position updates: callback(lat, lon)."""
        self._position_callbacks.append(callback)

    def register_fix_acquired_callback(self, callback) -> None:
        """Register callback for fix acquisition: callback(reading)."""
        self._fix_acquired_callbacks.append(callback)

    def register_fix_lost_callback(self, callback) -> None:
        """Register callback for fix loss: callback()."""
        self._fix_lost_callbacks.append(callback)

    @property
    def reading(self) -> GPSReading:
        """Get the latest GPS reading. Thread-safe."""
        with self._lock:
            return self._latest

    @property
    def position(self) -> tuple[float, float]:
        """Get current lat/lon tuple."""
        with self._lock:
            return (self._latest.lat, self._latest.lon)

    @property
    def has_fix(self) -> bool:
        with self._lock:
            return self._latest.has_fix

    @property
    def hardware_available(self) -> bool:
        return self._hardware_available

    @property
    def is_running(self) -> bool:
        return self._running

    def release(self) -> None:
        """Release GPS resources. Called during shutdown."""
        self.stop()
        if self._hardware_available and self._gpsd_session is not None:
            try:
                self._gpsd_session.close()
            except Exception as e:
                logger.error("GPS release error: %s", e)
            self._gpsd_session = None
        logger.info("GPS resources released")

    def __repr__(self) -> str:
        r: GPSReading = self.reading
        return (
            f"CerberusGPS(fix={'yes' if r.has_fix else 'no'}, "
            f"pos=({r.lat:.7f}, {r.lon:.7f}), "
            f"sats={r.satellites}, "
            f"hw={'yes' if self._hardware_available else 'sim'})"
        )