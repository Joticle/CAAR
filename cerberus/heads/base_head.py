"""
Cerberus Base Head — Abstract Base Class
All payload heads inherit from this class. Defines the lifecycle,
interface contract, and shared behavior that every head must implement.
Heads are hot-swappable mission modules — each gives Cerberus a
different mind for a different mission.
"""

import time
import logging
import threading
from typing import Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class HeadState(Enum):
    """Head lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class HeadInfo:
    """Metadata about a payload head."""
    name: str = ""
    description: str = ""
    version: str = "1.0"
    requires_camera: bool = False
    requires_gps: bool = False
    requires_sensors: bool = False
    requires_audio: bool = False
    requires_servos: bool = False
    supported_tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "requires_camera": self.requires_camera,
            "requires_gps": self.requires_gps,
            "requires_sensors": self.requires_sensors,
            "requires_audio": self.requires_audio,
            "requires_servos": self.requires_servos,
            "supported_tasks": self.supported_tasks
        }


@dataclass
class HeadStatus:
    """Current head operational status."""
    name: str = ""
    state: HeadState = HeadState.UNLOADED
    active_seconds: float = 0.0
    detections: int = 0
    errors: int = 0
    last_activity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "active_seconds": round(self.active_seconds, 1),
            "detections": self.detections,
            "errors": self.errors,
            "last_activity": self.last_activity,
            "metadata": self.metadata
        }


class BaseHead(ABC):
    """
    Abstract base class for all Cerberus payload heads.

    Lifecycle:
        1. __init__  — construct with config and subsystem references
        2. load()    — initialize head-specific hardware and models
        3. start()   — begin active operation (scanning, detecting, logging)
        4. pause()   — temporarily suspend operation
        5. resume()  — resume from pause
        6. stop()    — stop active operation
        7. unload()  — release all head-specific resources

    Every head must implement:
        - info property  — return HeadInfo describing capabilities
        - _on_load()     — head-specific initialization
        - _on_start()    — head-specific activation
        - _on_stop()     — head-specific deactivation
        - _on_unload()   — head-specific cleanup
        - _run_cycle()   — one iteration of the head's main work loop
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._state: HeadState = HeadState.UNLOADED
        self._lock: threading.Lock = threading.Lock()

        self._detection_count: int = 0
        self._error_count: int = 0
        self._start_time: float = 0.0
        self._last_activity: str = ""

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()

        self._cycle_interval: float = config.get(
            "heads", self.info.name, "cycle_interval", default=1.0
        ) if self.info.name else 1.0

        self._db: Optional[Any] = None
        self._mqtt: Optional[Any] = None
        self._camera: Optional[Any] = None
        self._gps: Optional[Any] = None
        self._environment: Optional[Any] = None
        self._classifier: Optional[Any] = None

    def bind_db(self, db: Any) -> None:
        """Bind database for detection and event logging."""
        self._db = db

    def bind_mqtt(self, mqtt_client: Any) -> None:
        """Bind MQTT for telemetry and alert publishing."""
        self._mqtt = mqtt_client

    def bind_camera(self, camera: Any) -> None:
        """Bind camera interface."""
        self._camera = camera

    def bind_gps(self, gps: Any) -> None:
        """Bind GPS interface."""
        self._gps = gps

    def bind_environment(self, environment: Any) -> None:
        """Bind environmental sensor manager."""
        self._environment = environment

    def bind_classifier(self, classifier: Any) -> None:
        """Bind the TFLite classifier manager."""
        self._classifier = classifier

    @property
    @abstractmethod
    def info(self) -> HeadInfo:
        """Return head metadata. Must be implemented by every head."""
        ...

    @abstractmethod
    def _on_load(self) -> bool:
        """
        Head-specific initialization.
        Load models, configure sensors, prepare resources.
        Return True on success, False on failure.
        """
        ...

    @abstractmethod
    def _on_start(self) -> bool:
        """
        Head-specific activation.
        Start cameras, begin sensor polling, arm detectors.
        Return True on success, False on failure.
        """
        ...

    @abstractmethod
    def _on_stop(self) -> None:
        """
        Head-specific deactivation.
        Stop cameras, halt sensor polling, disarm detectors.
        """
        ...

    @abstractmethod
    def _on_unload(self) -> None:
        """
        Head-specific cleanup.
        Release models, close files, free resources.
        """
        ...

    @abstractmethod
    def _run_cycle(self) -> None:
        """
        One iteration of the head's main work loop.
        Called repeatedly while the head is active.
        Perform scanning, detection, logging — whatever this head does.
        """
        ...

    def load(self) -> bool:
        """Initialize the head. Call after binding subsystems."""
        with self._lock:
            if self._state != HeadState.UNLOADED:
                logger.warning("Head '%s' already loaded (state=%s)", self.info.name, self._state.value)
                return False
            self._state = HeadState.LOADING

        logger.info("Loading head: %s", self.info.name)

        if not self._check_requirements():
            with self._lock:
                self._state = HeadState.ERROR
            return False

        try:
            success: bool = self._on_load()
            if success:
                with self._lock:
                    self._state = HeadState.READY
                logger.info("Head '%s' loaded and ready", self.info.name)
            else:
                with self._lock:
                    self._state = HeadState.ERROR
                logger.error("Head '%s' failed to load", self.info.name)
            return success

        except Exception as e:
            logger.error("Head '%s' load error: %s", self.info.name, e)
            with self._lock:
                self._state = HeadState.ERROR
                self._error_count += 1
            return False

    def start(self) -> bool:
        """Start active head operation."""
        with self._lock:
            if self._state != HeadState.READY:
                logger.warning(
                    "Cannot start head '%s' — state is %s (must be READY)",
                    self.info.name, self._state.value
                )
                return False

        logger.info("Starting head: %s", self.info.name)

        try:
            success: bool = self._on_start()
            if not success:
                logger.error("Head '%s' failed to start", self.info.name)
                return False
        except Exception as e:
            logger.error("Head '%s' start error: %s", self.info.name, e)
            with self._lock:
                self._error_count += 1
            return False

        self._running = True
        self._start_time = time.time()

        with self._lock:
            self._state = HeadState.ACTIVE

        self._thread = threading.Thread(
            target=self._work_loop,
            name=f"head-{self.info.name}",
            daemon=True
        )
        self._thread.start()

        logger.info("Head '%s' ACTIVE", self.info.name)
        return True

    def stop(self) -> None:
        """Stop active head operation."""
        if not self._running:
            return

        self._running = False
        self._pause_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Head '%s' thread did not stop cleanly", self.info.name)
            self._thread = None

        try:
            self._on_stop()
        except Exception as e:
            logger.error("Head '%s' stop error: %s", self.info.name, e)

        with self._lock:
            self._state = HeadState.READY

        logger.info("Head '%s' stopped", self.info.name)

    def unload(self) -> None:
        """Release all head resources."""
        if self._running:
            self.stop()

        with self._lock:
            self._state = HeadState.UNLOADING

        logger.info("Unloading head: %s", self.info.name)

        try:
            self._on_unload()
        except Exception as e:
            logger.error("Head '%s' unload error: %s", self.info.name, e)

        with self._lock:
            self._state = HeadState.UNLOADED

        logger.info("Head '%s' unloaded", self.info.name)

    def pause(self) -> None:
        """Pause the head's work loop."""
        with self._lock:
            if self._state != HeadState.ACTIVE:
                return
            self._state = HeadState.PAUSED

        self._pause_event.clear()
        logger.info("Head '%s' paused", self.info.name)

    def resume(self) -> None:
        """Resume the head's work loop."""
        with self._lock:
            if self._state != HeadState.PAUSED:
                return
            self._state = HeadState.ACTIVE

        self._pause_event.set()
        logger.info("Head '%s' resumed", self.info.name)

    def _work_loop(self) -> None:
        """Background work loop — calls _run_cycle repeatedly."""
        logger.info("Head '%s' work loop started", self.info.name)

        while self._running:
            self._pause_event.wait()
            if not self._running:
                break

            try:
                self._run_cycle()
            except Exception as e:
                logger.error("Head '%s' cycle error: %s", self.info.name, e)
                with self._lock:
                    self._error_count += 1

            for _ in range(int(self._cycle_interval * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

        logger.info("Head '%s' work loop stopped", self.info.name)

    def _check_requirements(self) -> bool:
        """Verify that all required subsystems are bound."""
        head_info: HeadInfo = self.info
        ok: bool = True

        if head_info.requires_camera and self._camera is None:
            logger.error("Head '%s' requires camera but none bound", head_info.name)
            ok = False

        if head_info.requires_gps and self._gps is None:
            logger.error("Head '%s' requires GPS but none bound", head_info.name)
            ok = False

        if head_info.requires_sensors and self._environment is None:
            logger.error("Head '%s' requires sensors but none bound", head_info.name)
            ok = False

        return ok

    def _record_detection(
        self,
        detection_type: str,
        label: str,
        confidence: float = 0.0,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Helper: record a detection to DB and MQTT."""
        with self._lock:
            self._detection_count += 1
            self._last_activity = f"Detection: {label} ({confidence:.1%})"

        gps_data: dict[str, float] = {}
        if self._gps is not None:
            reading = self._gps.reading
            if reading.has_fix:
                gps_data = {"lat": reading.lat, "lon": reading.lon}

        detection_data: dict[str, Any] = {
            "head": self.info.name,
            "type": detection_type,
            "label": label,
            "confidence": round(confidence, 4),
            "timestamp": time.time(),
            **gps_data,
            **(metadata or {})
        }

        if self._db is not None:
            try:
                self._db.log_detection(
                    detection_type=detection_type,
                    label=label,
                    confidence=confidence,
                    metadata=str(detection_data)
                )
            except Exception as e:
                logger.error("Failed to log detection: %s", e)

        if self._mqtt is not None:
            try:
                self._mqtt.publish_detection(detection_type, detection_data)
            except Exception:
                pass

    def _record_activity(self, activity: str) -> None:
        """Helper: update last activity string."""
        with self._lock:
            self._last_activity = activity

    @property
    def status(self) -> HeadStatus:
        """Get current head status."""
        with self._lock:
            active_time: float = 0.0
            if self._start_time > 0 and self._state == HeadState.ACTIVE:
                active_time = time.time() - self._start_time

            return HeadStatus(
                name=self.info.name,
                state=self._state,
                active_seconds=active_time,
                detections=self._detection_count,
                errors=self._error_count,
                last_activity=self._last_activity
            )

    @property
    def head_state(self) -> HeadState:
        with self._lock:
            return self._state

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._state == HeadState.ACTIVE

    @property
    def detection_count(self) -> int:
        with self._lock:
            return self._detection_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.info.name}', "
            f"state={self._state.value}, "
            f"detections={self._detection_count})"
        )