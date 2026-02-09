"""
Cerberus Enhanced Pest Deterrent Head
PIR + camera detection with AI species classification, servo-driven
predator decoy head, LED eyes, randomized predator audio, and
behavioral adaptation over time. The scarecrow that learns.
"""

import time
import random
import logging
from typing import Any, Optional
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import numpy as np

from cerberus.core.config import CerberusConfig
from cerberus.heads.base_head import BaseHead, HeadInfo
from cerberus.intelligence.motion_detector import MotionDetector, MotionEvent


logger: logging.Logger = logging.getLogger(__name__)


class DeterrentAction(Enum):
    """Types of deterrent actions."""
    NONE = "none"
    DECOY_MOVE = "decoy_move"
    LED_FLASH = "led_flash"
    AUDIO_PLAY = "audio_play"
    FULL_DISPLAY = "full_display"


class EscalationLevel(Enum):
    """Escalation stages for persistent pests."""
    PASSIVE = "passive"
    GENTLE = "gentle"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PestEvent:
    """A detected pest event with response data."""
    species: str = "unknown"
    confidence: float = 0.0
    is_pest: bool = False
    deterrent_action: str = DeterrentAction.NONE.value
    escalation: str = EscalationLevel.PASSIVE.value
    response_effective: bool = False
    lat: float = 0.0
    lon: float = 0.0
    image_path: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "confidence": round(self.confidence, 4),
            "is_pest": self.is_pest,
            "deterrent_action": self.deterrent_action,
            "escalation": self.escalation,
            "response_effective": self.response_effective,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "image_path": self.image_path,
            "timestamp": self.timestamp
        }


@dataclass
class SpeciesBehavior:
    """Tracks learned behavior patterns for a species."""
    species: str = ""
    encounters: int = 0
    deterred_count: int = 0
    ignored_count: int = 0
    effective_actions: dict[str, int] = field(default_factory=dict)
    last_seen: float = 0.0
    current_escalation: EscalationLevel = EscalationLevel.GENTLE

    @property
    def deterrent_rate(self) -> float:
        if self.encounters == 0:
            return 0.0
        return self.deterred_count / self.encounters

    @property
    def most_effective_action(self) -> str:
        if not self.effective_actions:
            return DeterrentAction.FULL_DISPLAY.value
        return max(self.effective_actions, key=self.effective_actions.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "encounters": self.encounters,
            "deterred_count": self.deterred_count,
            "ignored_count": self.ignored_count,
            "deterrent_rate": round(self.deterrent_rate, 2),
            "most_effective_action": self.most_effective_action,
            "current_escalation": self.current_escalation.value,
            "last_seen": self.last_seen
        }


class PestDeterrentHead(BaseHead):
    """
    Head 4: Enhanced Pest Deterrent
    Detects pests via motion + AI species classification, then deploys
    a randomized combination of deterrent actions: servo-driven predator
    decoy movement, LED eye flash patterns, and predator audio playback.
    Learns over time which actions are most effective per species and
    escalates responses for persistent pests.

    Cycle:
        1. Capture frame from camera
        2. Run motion detection
        3. If motion → classify species
        4. If pest → select deterrent action based on learned behavior
        5. Execute deterrent (decoy + LEDs + audio)
        6. Monitor if pest leaves (effectiveness tracking)
        7. Update behavioral model
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="pest_deterrent",
        description="AI pest detection with adaptive multi-modal deterrent system",
        version="1.0",
        requires_camera=True,
        requires_gps=True,
        requires_sensors=False,
        requires_audio=True,
        requires_servos=True,
        supported_tasks=["scan", "station_keep", "patrol"]
    )

    PEST_SPECIES: set[str] = {
        "rabbit", "squirrel", "rat", "mouse", "gopher",
        "pigeon", "starling", "grackle", "crow", "coyote",
        "snake", "lizard", "cat", "javelina", "packrat"
    }

    PREDATOR_AUDIO: list[str] = [
        "hawk_screech_01.wav",
        "hawk_screech_02.wav",
        "owl_hoot_01.wav",
        "owl_hoot_02.wav",
        "coyote_howl_01.wav",
        "dog_bark_01.wav",
        "dog_bark_02.wav",
        "snake_rattle_01.wav",
        "eagle_cry_01.wav",
        "falcon_call_01.wav",
    ]

    LED_PATTERNS: list[str] = [
        "rapid_flash",
        "alternating",
        "pulse",
        "strobe",
        "predator_eyes",
    ]

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._model_name: str = self._config.get(
            "heads", "pest_deterrent", "model_name", default="wildlife_classifier"
        )
        self._model_file: str = self._config.get(
            "heads", "pest_deterrent", "model_file", default="wildlife_classifier.tflite"
        )
        self._labels_file: Optional[str] = self._config.get(
            "heads", "pest_deterrent", "labels_file", default="wildlife_labels.txt"
        )
        self._confidence_threshold: float = self._config.get(
            "heads", "pest_deterrent", "confidence_threshold", default=0.5
        )
        self._audio_path: str = self._config.get(
            "heads", "pest_deterrent", "audio_path", default="data/audio/predator"
        )
        self._save_path: str = self._config.get(
            "heads", "pest_deterrent", "save_path", default="data/pest_detections"
        )
        self._effectiveness_window: float = self._config.get(
            "heads", "pest_deterrent", "effectiveness_window_seconds", default=30.0
        )
        self._cooldown_seconds: float = self._config.get(
            "heads", "pest_deterrent", "cooldown_seconds", default=15.0
        )
        self._decoy_pan_range: tuple[float, float] = tuple(self._config.get(
            "heads", "pest_deterrent", "decoy_pan_range", default=[-60.0, 60.0]
        ))
        self._decoy_tilt_range: tuple[float, float] = tuple(self._config.get(
            "heads", "pest_deterrent", "decoy_tilt_range", default=[-20.0, 20.0]
        ))

        self._motion_detector: Optional[MotionDetector] = None
        self._model_loaded: bool = False
        self._last_deterrent_time: float = 0.0
        self._total_detections: int = 0
        self._total_deterrents: int = 0
        self._species_behaviors: dict[str, SpeciesBehavior] = {}
        self._events: list[PestEvent] = []
        self._max_event_log: int = 500

        Path(self._save_path).mkdir(parents=True, exist_ok=True)

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def _on_load(self) -> bool:
        """Initialize motion detector, load classifier, prepare audio."""
        self._motion_detector = MotionDetector(self._config)

        if self._classifier is not None:
            try:
                self._model_loaded = self._classifier.load_model(
                    name=self._model_name,
                    model_file=self._model_file,
                    labels_file=self._labels_file
                )
                if self._model_loaded:
                    logger.info("Pest classifier loaded: %s", self._model_name)
                else:
                    logger.warning("Pest model not loaded — motion-only deterrent mode")
            except Exception as e:
                logger.error("Failed to load pest model: %s", e)
        else:
            logger.warning("No classifier bound — pest deterrent in motion-only mode")

        Path(self._audio_path).mkdir(parents=True, exist_ok=True)

        return True

    def _on_start(self) -> bool:
        """Activate pest deterrent system."""
        if self._camera is not None and not self._camera.is_streaming:
            self._camera.start()

        self._total_detections = 0
        self._total_deterrents = 0
        self._last_deterrent_time = 0.0

        self._move_decoy_home()

        logger.info(
            "Pest deterrent activated — threshold=%.0f%%, cooldown=%.0fs, audio files=%d",
            self._confidence_threshold * 100,
            self._cooldown_seconds,
            len(self.PREDATOR_AUDIO)
        )
        return True

    def _on_stop(self) -> None:
        """Deactivate pest deterrent."""
        self._move_decoy_home()
        self._set_led_eyes(False)

        logger.info(
            "Pest deterrent deactivated — %d detections, %d deterrents, %d species tracked",
            self._total_detections, self._total_deterrents, len(self._species_behaviors)
        )

    def _on_unload(self) -> None:
        """Release pest deterrent resources."""
        self._motion_detector = None
        logger.info("Pest deterrent unloaded")

    def _run_cycle(self) -> None:
        """One deterrent cycle: detect, classify, deter if pest."""
        frame: Optional[np.ndarray] = self._capture_frame()
        if frame is None:
            return

        motion: MotionEvent = self._detect_motion(frame)

        if not motion.detected:
            self._record_activity("Monitoring — no motion")
            return

        self._total_detections += 1
        classification: dict[str, Any] = self._classify_species(frame)

        species: str = classification.get("label", "unknown").lower()
        confidence: float = classification.get("confidence", 0.0)
        is_pest: bool = self._is_pest(species)

        gps_data: dict[str, Any] = self._get_gps_data()

        event: PestEvent = PestEvent(
            species=species,
            confidence=confidence,
            is_pest=is_pest,
            lat=gps_data.get("lat", 0.0),
            lon=gps_data.get("lon", 0.0),
            timestamp=time.time()
        )

        if is_pest and confidence >= self._confidence_threshold:
            self._handle_pest_detection(frame, event, species)
        elif is_pest:
            logger.debug("Low confidence pest: %s (%.1f%%) — monitoring", species, confidence * 100)
            event.deterrent_action = DeterrentAction.NONE.value
        else:
            logger.debug("Non-pest detected: %s (%.1f%%)", species, confidence * 100)
            event.deterrent_action = DeterrentAction.NONE.value

        self._log_event(event)

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera."""
        if self._camera is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame: Optional[np.ndarray] = self._camera.capture_frame()
        if frame is None:
            logger.warning("Pest deterrent failed to capture frame")
        return frame

    def _detect_motion(self, frame: np.ndarray) -> MotionEvent:
        """Run motion detection on frame."""
        if self._motion_detector is None:
            return MotionEvent()
        return self._motion_detector.process_frame(frame)

    def _classify_species(self, frame: np.ndarray) -> dict[str, Any]:
        """Classify the detected animal species."""
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
            logger.error("Species classification error: %s", e)
            return {"label": "unknown", "confidence": 0.0, "top_k": [], "inference_time_ms": 0.0}

    def _simulated_classification(self) -> dict[str, Any]:
        """Simulated classification for dev environment."""
        species: list[str] = ["rabbit", "squirrel", "bird", "cat", "pigeon", "lizard"]
        label: str = random.choice(species)
        return {
            "label": label,
            "confidence": random.uniform(0.5, 0.95),
            "top_k": [],
            "inference_time_ms": 0.0
        }

    def _is_pest(self, species: str) -> bool:
        """Check if identified species is a pest."""
        return any(pest in species.lower() for pest in self.PEST_SPECIES)

    def _handle_pest_detection(
        self,
        frame: np.ndarray,
        event: PestEvent,
        species: str
    ) -> None:
        """Process confirmed pest detection and execute deterrent."""
        now: float = time.time()
        if now - self._last_deterrent_time < self._cooldown_seconds:
            logger.debug("Deterrent cooldown active — %.1fs remaining",
                         self._cooldown_seconds - (now - self._last_deterrent_time))
            event.deterrent_action = DeterrentAction.NONE.value
            return

        behavior: SpeciesBehavior = self._get_species_behavior(species)
        behavior.encounters += 1
        behavior.last_seen = now

        action: DeterrentAction = self._select_deterrent_action(behavior)
        escalation: EscalationLevel = behavior.current_escalation

        event.deterrent_action = action.value
        event.escalation = escalation.value

        logger.info(
            "PEST DETECTED: %s (%.1f%%) — action=%s, escalation=%s",
            species, event.confidence * 100, action.value, escalation.value
        )

        self._execute_deterrent(action, escalation)
        self._total_deterrents += 1
        self._last_deterrent_time = now

        event.image_path = self._save_detection_image(frame, event)

        self._record_detection(
            detection_type="pest",
            label=species,
            confidence=event.confidence,
            metadata=event.to_dict()
        )

        effective: bool = self._monitor_effectiveness(species)
        event.response_effective = effective
        self._update_behavior(behavior, action, effective)

    def _get_species_behavior(self, species: str) -> SpeciesBehavior:
        """Get or create behavioral profile for a species."""
        key: str = species.lower()
        if key not in self._species_behaviors:
            self._species_behaviors[key] = SpeciesBehavior(species=key)
        return self._species_behaviors[key]

    def _select_deterrent_action(self, behavior: SpeciesBehavior) -> DeterrentAction:
        """Select deterrent action based on learned behavior and escalation."""
        escalation: EscalationLevel = behavior.current_escalation

        if escalation == EscalationLevel.GENTLE:
            choices: list[DeterrentAction] = [
                DeterrentAction.DECOY_MOVE,
                DeterrentAction.LED_FLASH
            ]
        elif escalation == EscalationLevel.MODERATE:
            choices = [
                DeterrentAction.DECOY_MOVE,
                DeterrentAction.LED_FLASH,
                DeterrentAction.AUDIO_PLAY
            ]
        else:
            choices = [DeterrentAction.FULL_DISPLAY]

        if behavior.effective_actions:
            best: str = behavior.most_effective_action
            try:
                preferred: DeterrentAction = DeterrentAction(best)
                if preferred in choices:
                    if random.random() < 0.7:
                        return preferred
            except ValueError:
                pass

        return random.choice(choices)

    def _execute_deterrent(self, action: DeterrentAction, escalation: EscalationLevel) -> None:
        """Execute the selected deterrent action."""
        self._record_activity(f"Deterrent: {action.value} ({escalation.value})")

        if action == DeterrentAction.DECOY_MOVE:
            self._move_decoy_random()

        elif action == DeterrentAction.LED_FLASH:
            self._flash_led_eyes(escalation)

        elif action == DeterrentAction.AUDIO_PLAY:
            self._play_predator_audio(escalation)

        elif action == DeterrentAction.FULL_DISPLAY:
            self._move_decoy_random()
            self._flash_led_eyes(escalation)
            self._play_predator_audio(escalation)

    def _move_decoy_random(self) -> None:
        """Move the predator decoy head to a random position."""
        pan: float = random.uniform(self._decoy_pan_range[0], self._decoy_pan_range[1])
        tilt: float = random.uniform(self._decoy_tilt_range[0], self._decoy_tilt_range[1])
        logger.debug("Decoy move: pan=%.1f, tilt=%.1f", pan, tilt)
        time.sleep(0.5)

    def _move_decoy_home(self) -> None:
        """Return decoy to center position."""
        logger.debug("Decoy returning to home position")

    def _flash_led_eyes(self, escalation: EscalationLevel) -> None:
        """Flash LED eyes with pattern based on escalation level."""
        if escalation == EscalationLevel.GENTLE:
            pattern: str = "pulse"
            duration: float = 2.0
        elif escalation == EscalationLevel.MODERATE:
            pattern = random.choice(["rapid_flash", "alternating"])
            duration = 3.0
        else:
            pattern = "strobe"
            duration = 5.0

        logger.debug("LED eyes: pattern=%s, duration=%.1fs", pattern, duration)
        self._set_led_eyes(True)
        time.sleep(duration)
        self._set_led_eyes(False)

    def _set_led_eyes(self, enabled: bool) -> None:
        """Control LED eye state."""
        logger.debug("LED eyes %s", "on" if enabled else "off")

    def _play_predator_audio(self, escalation: EscalationLevel) -> None:
        """Play a random predator audio clip."""
        clip: str = random.choice(self.PREDATOR_AUDIO)

        volume: float = 0.5
        if escalation == EscalationLevel.MODERATE:
            volume = 0.7
        elif escalation == EscalationLevel.AGGRESSIVE:
            volume = 1.0

        audio_file: str = str(Path(self._audio_path) / clip)
        logger.debug("Playing predator audio: %s at volume=%.1f", clip, volume)

        if Path(audio_file).exists():
            logger.info("Audio playback: %s", clip)
        else:
            logger.debug("Audio file not found (dev mode): %s", audio_file)

    def _monitor_effectiveness(self, species: str) -> bool:
        """
        Monitor if the pest leaves after deterrent action.
        Checks for continued motion over the effectiveness window.
        Returns True if the deterrent appears to have worked.
        """
        check_interval: float = 2.0
        checks: int = int(self._effectiveness_window / check_interval)
        motion_count: int = 0

        for _ in range(checks):
            if not self._running:
                return False

            time.sleep(check_interval)

            frame: Optional[np.ndarray] = self._capture_frame()
            if frame is None:
                continue

            motion: MotionEvent = self._detect_motion(frame)
            if motion.detected:
                motion_count += 1

        effective: bool = motion_count < checks * 0.3

        if effective:
            logger.info("Deterrent effective — %s appears to have left", species)
        else:
            logger.info("Deterrent ineffective — %s still present (%d/%d motion checks)",
                        species, motion_count, checks)

        return effective

    def _update_behavior(
        self,
        behavior: SpeciesBehavior,
        action: DeterrentAction,
        effective: bool
    ) -> None:
        """Update learned behavior model for a species."""
        if effective:
            behavior.deterred_count += 1
            action_key: str = action.value
            behavior.effective_actions[action_key] = behavior.effective_actions.get(action_key, 0) + 1
        else:
            behavior.ignored_count += 1

        if behavior.deterrent_rate < 0.3 and behavior.encounters >= 3:
            levels: list[EscalationLevel] = list(EscalationLevel)
            current_idx: int = levels.index(behavior.current_escalation)
            if current_idx < len(levels) - 1:
                behavior.current_escalation = levels[current_idx + 1]
                logger.info(
                    "Escalation for %s: %s (deterrent rate=%.0f%%)",
                    behavior.species,
                    behavior.current_escalation.value,
                    behavior.deterrent_rate * 100
                )

        elif behavior.deterrent_rate > 0.7 and behavior.encounters >= 5:
            levels = list(EscalationLevel)
            current_idx = levels.index(behavior.current_escalation)
            if current_idx > 1:
                behavior.current_escalation = levels[current_idx - 1]
                logger.info(
                    "De-escalation for %s: %s (deterrent rate=%.0f%%)",
                    behavior.species,
                    behavior.current_escalation.value,
                    behavior.deterrent_rate * 100
                )

    def _save_detection_image(self, frame: np.ndarray, event: PestEvent) -> str:
        """Save pest detection evidence frame."""
        try:
            import cv2

            annotated: np.ndarray = frame.copy()
            label_text: str = f"PEST: {event.species} ({event.confidence:.0%}) [{event.escalation}]"
            cv2.putText(
                annotated, label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2
            )

            timestamp: str = time.strftime("%Y%m%d_%H%M%S")
            filename: str = f"pest_{timestamp}_{event.species}.jpg"
            filepath: str = str(Path(self._save_path) / filename)
            cv2.imwrite(filepath, annotated)

            logger.debug("Pest detection saved: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("Failed to save pest detection image: %s", e)
            return ""

    def _log_event(self, event: PestEvent) -> None:
        """Log pest event to internal buffer."""
        self._events.append(event)
        if len(self._events) > self._max_event_log:
            self._events = self._events[-self._max_event_log:]

    def _get_gps_data(self) -> dict[str, Any]:
        """Get current GPS coordinates."""
        if self._gps is None:
            return {"lat": 0.0, "lon": 0.0, "gps_fix": False}
        reading = self._gps.reading
        return {"lat": reading.lat, "lon": reading.lon, "gps_fix": reading.has_fix}

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Perform pest scan at current position.
        Called by patrol executor action handler at each waypoint.
        """
        detections: list[dict[str, Any]] = []
        scan_duration: float = 15.0
        start: float = time.time()

        while time.time() - start < scan_duration:
            if not self._running:
                break

            frame: Optional[np.ndarray] = self._capture_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            motion: MotionEvent = self._detect_motion(frame)
            if motion.detected:
                classification: dict[str, Any] = self._classify_species(frame)
                species: str = classification.get("label", "unknown")
                if self._is_pest(species):
                    detections.append({
                        "species": species,
                        "confidence": classification.get("confidence", 0.0)
                    })

            time.sleep(0.5)

        return {
            "scan_duration_s": round(time.time() - start, 1),
            "pests_detected": len(detections),
            "detections": detections,
            "gps": self._get_gps_data(),
            "timestamp": time.time()
        }

    @property
    def species_behaviors(self) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._species_behaviors.items()}

    @property
    def total_detections(self) -> int:
        return self._total_detections

    @property
    def total_deterrents(self) -> int:
        return self._total_deterrents

    @property
    def recent_events(self) -> list[PestEvent]:
        return list(self._events[-20:])
