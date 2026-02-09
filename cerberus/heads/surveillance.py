"""
Cerberus Surveillance / Patrol Cam Head
Wide-angle Pi Camera 3 + IR LEDs + AI-driven motion detection +
threat classification + autonomous patrol routes + live MJPEG streaming.
The sentry mode head — Cerberus watches, classifies, and alerts.
"""

import time
import logging
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from cerberus.core.config import CerberusConfig
from cerberus.heads.base_head import BaseHead, HeadInfo
from cerberus.intelligence.motion_detector import MotionDetector, MotionEvent


logger: logging.Logger = logging.getLogger(__name__)


class ThreatLevel:
    """Threat classification levels."""
    NONE: str = "none"
    LOW: str = "low"
    MEDIUM: str = "medium"
    HIGH: str = "high"
    CRITICAL: str = "critical"


@dataclass
class SurveillanceEvent:
    """A detected surveillance event."""
    event_type: str = "motion"
    threat_level: str = ThreatLevel.NONE
    label: str = "unknown"
    confidence: float = 0.0
    region_count: int = 0
    motion_pct: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    image_path: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "threat_level": self.threat_level,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "region_count": self.region_count,
            "motion_pct": round(self.motion_pct, 2),
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "image_path": self.image_path,
            "timestamp": self.timestamp
        }


class SurveillanceHead(BaseHead):
    """
    Head 2: Surveillance / Patrol Cam
    Monitors for motion, classifies detected objects, determines
    threat level, saves evidence frames, and publishes alerts.
    Optimized for sentry mode — stationary overnight surveillance
    or slow patrol with continuous monitoring.

    Cycle:
        1. Capture frame from camera
        2. Run through motion detection pipeline
        3. If motion detected → classify the moving object
        4. Assess threat level based on classification
        5. Save evidence frame if threat is medium+
        6. Publish alert to MQTT for Dashboard notification
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="surveillance",
        description="AI-driven motion detection, threat classification, and patrol monitoring",
        version="1.0",
        requires_camera=True,
        requires_gps=True,
        requires_sensors=False,
        requires_audio=False,
        requires_servos=False,
        supported_tasks=["scan", "patrol", "station_keep"]
    )

    THREAT_LABELS: dict[str, str] = {
        "person": ThreatLevel.HIGH,
        "human": ThreatLevel.HIGH,
        "car": ThreatLevel.MEDIUM,
        "vehicle": ThreatLevel.MEDIUM,
        "truck": ThreatLevel.MEDIUM,
        "dog": ThreatLevel.LOW,
        "cat": ThreatLevel.LOW,
        "coyote": ThreatLevel.MEDIUM,
        "bird": ThreatLevel.NONE,
        "rabbit": ThreatLevel.NONE,
        "squirrel": ThreatLevel.NONE,
    }

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._model_name: str = self._config.get(
            "heads", "surveillance", "model_name", default="threat_classifier"
        )
        self._model_file: str = self._config.get(
            "heads", "surveillance", "model_file", default="threat_classifier.tflite"
        )
        self._labels_file: Optional[str] = self._config.get(
            "heads", "surveillance", "labels_file", default="threat_labels.txt"
        )
        self._confidence_threshold: float = self._config.get(
            "heads", "surveillance", "confidence_threshold", default=0.5
        )
        self._save_evidence: bool = self._config.get(
            "heads", "surveillance", "save_evidence", default=True
        )
        self._save_path: str = self._config.get(
            "heads", "surveillance", "save_path", default="data/surveillance"
        )
        self._alert_cooldown: float = self._config.get(
            "heads", "surveillance", "alert_cooldown_seconds", default=30.0
        )
        self._save_all_motion: bool = self._config.get(
            "heads", "surveillance", "save_all_motion", default=False
        )
        self._ir_enabled: bool = self._config.get(
            "heads", "surveillance", "ir_leds_enabled", default=True
        )

        self._motion_detector: Optional[MotionDetector] = None
        self._model_loaded: bool = False
        self._last_alert_time: float = 0.0
        self._total_motion_events: int = 0
        self._total_threats: int = 0
        self._events: list[SurveillanceEvent] = []
        self._max_event_log: int = 500

        Path(self._save_path).mkdir(parents=True, exist_ok=True)

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def _on_load(self) -> bool:
        """Initialize motion detector and load threat classifier."""
        self._motion_detector = MotionDetector(self._config)

        self._motion_detector.register_motion_start_callback(self._on_motion_start)
        self._motion_detector.register_motion_stop_callback(self._on_motion_stop)

        if self._classifier is not None:
            try:
                self._model_loaded = self._classifier.load_model(
                    name=self._model_name,
                    model_file=self._model_file,
                    labels_file=self._labels_file
                )
                if self._model_loaded:
                    logger.info("Threat classifier loaded: %s", self._model_name)
                else:
                    logger.warning("Threat model not loaded — motion-only mode")
            except Exception as e:
                logger.error("Failed to load threat model: %s", e)
        else:
            logger.warning("No classifier bound — surveillance in motion-only mode")

        return True

    def _on_start(self) -> bool:
        """Start camera and motion detection."""
        if self._camera is not None and not self._camera.is_streaming:
            self._camera.start()

        self._total_motion_events = 0
        self._total_threats = 0
        self._last_alert_time = 0.0

        if self._ir_enabled:
            self._set_ir_leds(True)

        logger.info(
            "Surveillance activated — threshold=%.0f%%, cooldown=%.0fs, IR=%s",
            self._confidence_threshold * 100,
            self._alert_cooldown,
            "on" if self._ir_enabled else "off"
        )
        return True

    def _on_stop(self) -> None:
        """Stop surveillance."""
        if self._ir_enabled:
            self._set_ir_leds(False)

        logger.info(
            "Surveillance deactivated — %d motion events, %d threats detected",
            self._total_motion_events, self._total_threats
        )

    def _on_unload(self) -> None:
        """Release surveillance resources."""
        self._motion_detector = None
        logger.info("Surveillance head unloaded")

    def _run_cycle(self) -> None:
        """One surveillance cycle: capture, detect motion, classify if needed."""
        frame: Optional[np.ndarray] = self._capture_frame()
        if frame is None:
            return

        motion_event: MotionEvent = self._detect_motion(frame)

        if motion_event.detected:
            self._handle_motion(frame, motion_event)

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if self._camera is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame: Optional[np.ndarray] = self._camera.capture_frame()
        if frame is None:
            logger.warning("Surveillance failed to capture frame")
        return frame

    def _detect_motion(self, frame: np.ndarray) -> MotionEvent:
        """Run frame through motion detection pipeline."""
        if self._motion_detector is None:
            return MotionEvent()

        return self._motion_detector.process_frame(frame)

    def _handle_motion(self, frame: np.ndarray, motion: MotionEvent) -> None:
        """Process a motion detection event."""
        self._total_motion_events += 1
        self._record_activity(
            f"Motion: {motion.region_count} regions, {motion.motion_pct:.1f}% area"
        )

        classification: dict[str, Any] = self._classify_motion(frame)
        threat_level: str = self._assess_threat(classification, motion)

        event: SurveillanceEvent = SurveillanceEvent(
            event_type="motion",
            threat_level=threat_level,
            label=classification.get("label", "unknown"),
            confidence=classification.get("confidence", 0.0),
            region_count=motion.region_count,
            motion_pct=motion.motion_pct,
            timestamp=time.time()
        )

        gps_data: dict[str, Any] = self._get_gps_data()
        event.lat = gps_data.get("lat", 0.0)
        event.lon = gps_data.get("lon", 0.0)

        if self._save_evidence and threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            event.image_path = self._save_evidence_frame(frame, motion, classification, threat_level)

        if self._save_all_motion and not event.image_path:
            event.image_path = self._save_evidence_frame(frame, motion, classification, threat_level)

        self._log_event(event)

        if threat_level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            self._total_threats += 1
            self._send_alert(event)

    def _classify_motion(self, frame: np.ndarray) -> dict[str, Any]:
        """Classify what caused the motion."""
        if self._classifier is None or not self._model_loaded:
            return self._simulated_classification()

        try:
            result = self._classifier.classify(
                model_name=self._model_name,
                image=frame,
                top_k=3,
                threshold=self._confidence_threshold
            )

            return {
                "label": result.label,
                "confidence": result.confidence,
                "top_k": result.top_k,
                "inference_time_ms": result.inference_time_ms
            }

        except Exception as e:
            logger.error("Threat classification error: %s", e)
            return {"label": "unknown", "confidence": 0.0, "top_k": [], "inference_time_ms": 0.0}

    def _simulated_classification(self) -> dict[str, Any]:
        """Return simulated classification for dev environment."""
        import random
        labels: list[str] = ["cat", "dog", "person", "bird", "squirrel", "unknown"]
        label: str = random.choice(labels)
        return {
            "label": label,
            "confidence": random.uniform(0.4, 0.95),
            "top_k": [],
            "inference_time_ms": 0.0
        }

    def _assess_threat(self, classification: dict[str, Any], motion: MotionEvent) -> str:
        """Determine threat level from classification and motion characteristics."""
        label: str = classification.get("label", "unknown").lower()
        confidence: float = classification.get("confidence", 0.0)

        if confidence < self._confidence_threshold:
            if motion.motion_pct > 10.0:
                return ThreatLevel.MEDIUM
            if motion.largest_area > 50000:
                return ThreatLevel.LOW
            return ThreatLevel.NONE

        for keyword, level in self.THREAT_LABELS.items():
            if keyword in label:
                return level

        if motion.motion_pct > 20.0:
            return ThreatLevel.MEDIUM

        return ThreatLevel.LOW

    def _save_evidence_frame(
        self,
        frame: np.ndarray,
        motion: MotionEvent,
        classification: dict[str, Any],
        threat_level: str
    ) -> str:
        """Save annotated evidence frame to disk."""
        try:
            import cv2

            annotated: np.ndarray = frame.copy()

            if self._motion_detector is not None:
                annotated = self._motion_detector.annotate_frame(annotated, motion)

            label: str = classification.get("label", "unknown")
            conf: float = classification.get("confidence", 0.0)
            threat_text: str = f"THREAT: {threat_level.upper()} | {label} ({conf:.0%})"

            color: tuple[int, int, int] = (0, 255, 0)
            if threat_level == ThreatLevel.HIGH or threat_level == ThreatLevel.CRITICAL:
                color = (0, 0, 255)
            elif threat_level == ThreatLevel.MEDIUM:
                color = (0, 165, 255)

            cv2.putText(
                annotated, threat_text,
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2
            )

            timestamp: str = time.strftime("%Y%m%d_%H%M%S")
            filename: str = f"surv_{timestamp}_{threat_level}_{label}.jpg"
            filepath: str = str(Path(self._save_path) / filename)

            cv2.imwrite(filepath, annotated)
            logger.debug("Evidence saved: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("Failed to save evidence frame: %s", e)
            return ""

    def _send_alert(self, event: SurveillanceEvent) -> None:
        """Send alert to MQTT if cooldown has elapsed."""
        now: float = time.time()
        if now - self._last_alert_time < self._alert_cooldown:
            return

        self._last_alert_time = now

        logger.warning(
            "SURVEILLANCE ALERT: %s threat — %s (%.1f%%) at (%.7f, %.7f)",
            event.threat_level, event.label, event.confidence * 100,
            event.lat, event.lon
        )

        self._record_detection(
            detection_type="threat",
            label=event.label,
            confidence=event.confidence,
            metadata=event.to_dict()
        )

    def _log_event(self, event: SurveillanceEvent) -> None:
        """Log surveillance event to internal buffer."""
        self._events.append(event)
        if len(self._events) > self._max_event_log:
            self._events = self._events[-self._max_event_log:]

    def _on_motion_start(self) -> None:
        """Callback: motion detection started."""
        self._record_activity("Motion detected — scanning")

    def _on_motion_stop(self) -> None:
        """Callback: motion detection stopped."""
        self._record_activity("Monitoring — no motion")

    def _set_ir_leds(self, enabled: bool) -> None:
        """Control IR LED illumination for night vision."""
        logger.info("IR LEDs %s", "enabled" if enabled else "disabled")

    def _get_gps_data(self) -> dict[str, Any]:
        """Get current GPS coordinates."""
        if self._gps is None:
            return {"lat": 0.0, "lon": 0.0, "gps_fix": False}

        reading = self._gps.reading
        return {
            "lat": reading.lat,
            "lon": reading.lon,
            "gps_fix": reading.has_fix
        }

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Perform surveillance scan at the current position.
        Called by patrol executor action handler at each waypoint.
        """
        scan_duration: float = 10.0
        events_detected: list[dict[str, Any]] = []
        start: float = time.time()

        while time.time() - start < scan_duration:
            frame: Optional[np.ndarray] = self._capture_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            motion: MotionEvent = self._detect_motion(frame)
            if motion.detected:
                classification: dict[str, Any] = self._classify_motion(frame)
                threat: str = self._assess_threat(classification, motion)
                events_detected.append({
                    "label": classification.get("label", "unknown"),
                    "confidence": classification.get("confidence", 0.0),
                    "threat_level": threat,
                    "motion_pct": motion.motion_pct
                })

            time.sleep(0.2)

        return {
            "scan_duration_s": round(time.time() - start, 1),
            "events_detected": len(events_detected),
            "events": events_detected,
            "gps": self._get_gps_data(),
            "timestamp": time.time()
        }

    @property
    def recent_events(self) -> list[SurveillanceEvent]:
        return list(self._events[-20:])

    @property
    def total_motion_events(self) -> int:
        return self._total_motion_events

    @property
    def total_threats(self) -> int:
        return self._total_threats

    @property
    def motion_active(self) -> bool:
        if self._motion_detector is None:
            return False
        return self._motion_detector.is_in_motion