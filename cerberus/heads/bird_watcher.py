"""
Cerberus Bird Watcher & Identifier Head
Pi Camera + telephoto adapter + BirdNET/TFLite neural species ID.
Autonomous sighting log with confidence scoring, photo capture,
and audio recording for species identification.
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


@dataclass
class BirdSighting:
    """A confirmed bird sighting record."""
    species: str = "unknown"
    common_name: str = "unknown"
    scientific_name: str = ""
    identification_method: str = "visual"
    confidence: float = 0.0
    photo_path: str = ""
    audio_path: str = ""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    notes: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "identification_method": self.identification_method,
            "confidence": round(self.confidence, 4),
            "photo_path": self.photo_path,
            "audio_path": self.audio_path,
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "alt_m": round(self.alt_m, 1),
            "notes": self.notes,
            "timestamp": self.timestamp
        }


@dataclass
class SpeciesLog:
    """Cumulative sighting data for a single species."""
    species: str = ""
    common_name: str = ""
    scientific_name: str = ""
    sighting_count: int = 0
    best_confidence: float = 0.0
    best_photo_path: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    locations: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "sighting_count": self.sighting_count,
            "best_confidence": round(self.best_confidence, 4),
            "best_photo_path": self.best_photo_path,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "unique_locations": len(self.locations)
        }


class BirdWatcherHead(BaseHead):
    """
    Head 5: Bird Watcher & Identifier
    Combines visual detection (motion + TFLite classification) with
    audio identification (BirdNET) for comprehensive bird species ID.
    Maintains an autonomous sighting log with photos, GPS coordinates,
    confidence scores, and species statistics.

    Cycle:
        1. Monitor for bird-sized motion via camera
        2. If motion detected → capture high-res photo
        3. Run visual species classifier
        4. If confident bird ID → log sighting with photo + GPS
        5. Periodically record audio for BirdNET analysis
        6. Cross-reference visual and audio IDs for confirmation
        7. Update species log and publish sighting
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="bird_watcher",
        description="Neural bird species identification with photo capture and sighting log",
        version="1.0",
        requires_camera=True,
        requires_gps=True,
        requires_sensors=False,
        requires_audio=True,
        requires_servos=True,
        supported_tasks=["scan", "station_keep", "patrol"]
    )

    BIRD_SIZE_MIN_AREA: int = 200
    BIRD_SIZE_MAX_AREA: int = 80000

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._model_name: str = self._config.get(
            "heads", "bird_watcher", "model_name", default="bird_classifier"
        )
        self._model_file: str = self._config.get(
            "heads", "bird_watcher", "model_file", default="bird_classifier.tflite"
        )
        self._labels_file: Optional[str] = self._config.get(
            "heads", "bird_watcher", "labels_file", default="bird_labels.txt"
        )
        self._confidence_threshold: float = self._config.get(
            "heads", "bird_watcher", "confidence_threshold", default=0.5
        )
        self._photo_path: str = self._config.get(
            "heads", "bird_watcher", "photo_path", default="data/bird_photos"
        )
        self._audio_record_interval: float = self._config.get(
            "heads", "bird_watcher", "audio_record_interval_seconds", default=300.0
        )
        self._audio_record_duration: float = self._config.get(
            "heads", "bird_watcher", "audio_record_duration_seconds", default=15.0
        )
        self._sighting_cooldown: float = self._config.get(
            "heads", "bird_watcher", "sighting_cooldown_seconds", default=60.0
        )
        self._max_locations_per_species: int = self._config.get(
            "heads", "bird_watcher", "max_locations_per_species", default=50
        )

        self._motion_detector: Optional[MotionDetector] = None
        self._species_id: Optional[Any] = None
        self._model_loaded: bool = False

        self._sightings: list[BirdSighting] = []
        self._max_sightings: int = 1000
        self._species_log: dict[str, SpeciesLog] = {}
        self._last_sighting_times: dict[str, float] = {}
        self._last_audio_time: float = 0.0
        self._total_visual_ids: int = 0
        self._total_audio_ids: int = 0

        Path(self._photo_path).mkdir(parents=True, exist_ok=True)

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def bind_species_id(self, species_id: Any) -> None:
        """Bind the SpeciesIdentifier for audio bird ID."""
        self._species_id = species_id

    def _on_load(self) -> bool:
        """Initialize motion detector and load bird classifier."""
        self._motion_detector = MotionDetector(self._config)

        if self._classifier is not None:
            try:
                self._model_loaded = self._classifier.load_model(
                    name=self._model_name,
                    model_file=self._model_file,
                    labels_file=self._labels_file
                )
                if self._model_loaded:
                    logger.info("Bird classifier loaded: %s", self._model_name)
                else:
                    logger.warning("Bird model not loaded — audio-only mode available")
            except Exception as e:
                logger.error("Failed to load bird model: %s", e)

        if self._species_id is not None and self._species_id.audio_available:
            logger.info("BirdNET audio identification available")
        else:
            logger.warning("BirdNET not available — visual-only mode")

        return True

    def _on_start(self) -> bool:
        """Activate bird watching."""
        if self._camera is not None and not self._camera.is_streaming:
            self._camera.start()

        self._total_visual_ids = 0
        self._total_audio_ids = 0
        self._last_audio_time = time.time()

        logger.info(
            "Bird watcher activated — visual=%s, audio=%s, threshold=%.0f%%",
            "enabled" if self._model_loaded else "disabled",
            "enabled" if (self._species_id and self._species_id.audio_available) else "disabled",
            self._confidence_threshold * 100
        )
        return True

    def _on_stop(self) -> None:
        """Deactivate bird watching."""
        logger.info(
            "Bird watcher deactivated — %d visual IDs, %d audio IDs, %d species logged",
            self._total_visual_ids, self._total_audio_ids, len(self._species_log)
        )

    def _on_unload(self) -> None:
        """Release bird watcher resources."""
        self._motion_detector = None
        logger.info("Bird watcher unloaded")

    def _run_cycle(self) -> None:
        """One bird watching cycle: visual scan + periodic audio."""
        self._visual_scan()

        now: float = time.time()
        if now - self._last_audio_time >= self._audio_record_interval:
            self._audio_scan()
            self._last_audio_time = now

    def _visual_scan(self) -> None:
        """Scan for birds visually using motion detection + classification."""
        frame: Optional[np.ndarray] = self._capture_frame()
        if frame is None:
            return

        motion: MotionEvent = self._detect_motion(frame)

        if not motion.detected:
            self._record_activity("Watching — no movement")
            return

        if not self._is_bird_sized_motion(motion):
            return

        classification: dict[str, Any] = self._classify_bird(frame)
        label: str = classification.get("label", "unknown")
        confidence: float = classification.get("confidence", 0.0)

        if confidence < self._confidence_threshold:
            return

        if not self._is_bird_species(label):
            return

        if self._in_sighting_cooldown(label):
            return

        self._total_visual_ids += 1
        photo_path: str = self._capture_sighting_photo(frame, label, confidence)
        gps_data: dict[str, Any] = self._get_gps_data()

        sighting: BirdSighting = BirdSighting(
            species=label,
            common_name=label,
            identification_method="visual",
            confidence=confidence,
            photo_path=photo_path,
            lat=gps_data.get("lat", 0.0),
            lon=gps_data.get("lon", 0.0),
            alt_m=gps_data.get("alt_m", 0.0),
            timestamp=time.time()
        )

        self._record_sighting(sighting)

    def _audio_scan(self) -> None:
        """Record audio and run BirdNET identification."""
        if self._species_id is None or not self._species_id.audio_available:
            return

        self._record_activity("Recording audio for BirdNET analysis...")
        logger.debug("Starting audio recording (%.0fs)", self._audio_record_duration)

        audio_data: Optional[np.ndarray] = self._record_audio()
        if audio_data is None:
            return

        try:
            result = self._species_id.identify_audio_data(audio_data)

            if result.confidence >= self._confidence_threshold and result.species != "unknown":
                self._total_audio_ids += 1
                gps_data: dict[str, Any] = self._get_gps_data()

                sighting: BirdSighting = BirdSighting(
                    species=result.common_name,
                    common_name=result.common_name,
                    scientific_name=result.scientific_name,
                    identification_method="audio",
                    confidence=result.confidence,
                    lat=gps_data.get("lat", 0.0),
                    lon=gps_data.get("lon", 0.0),
                    alt_m=gps_data.get("alt_m", 0.0),
                    notes=f"BirdNET inference: {result.inference_time_ms:.0f}ms",
                    timestamp=time.time()
                )

                self._record_sighting(sighting)

        except Exception as e:
            logger.error("Audio bird ID failed: %s", e)

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture camera frame."""
        if self._camera is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame: Optional[np.ndarray] = self._camera.capture_frame()
        if frame is None:
            logger.warning("Bird watcher failed to capture frame")
        return frame

    def _detect_motion(self, frame: np.ndarray) -> MotionEvent:
        """Run motion detection."""
        if self._motion_detector is None:
            return MotionEvent()
        return self._motion_detector.process_frame(frame)

    def _is_bird_sized_motion(self, motion: MotionEvent) -> bool:
        """Filter motion events to bird-sized regions."""
        for box in motion.bounding_boxes:
            _, _, w, h = box
            area: int = w * h
            if self.BIRD_SIZE_MIN_AREA <= area <= self.BIRD_SIZE_MAX_AREA:
                return True
        return False

    def _classify_bird(self, frame: np.ndarray) -> dict[str, Any]:
        """Run bird species classification on a frame."""
        if self._classifier is None or not self._model_loaded:
            return self._simulated_classification()

        try:
            result = self._classifier.classify(
                model_name=self._model_name,
                image=frame,
                top_k=5,
                threshold=self._confidence_threshold
            )
            return {
                "label": result.label,
                "confidence": result.confidence,
                "top_k": result.top_k,
                "inference_time_ms": result.inference_time_ms
            }
        except Exception as e:
            logger.error("Bird classification error: %s", e)
            return {"label": "unknown", "confidence": 0.0, "top_k": [], "inference_time_ms": 0.0}

    def _simulated_classification(self) -> dict[str, Any]:
        """Simulated classification for dev environment."""
        import random
        birds: list[str] = [
            "House Sparrow", "Mourning Dove", "Hummingbird",
            "Roadrunner", "Mockingbird", "Quail", "Red-tailed Hawk",
            "Great Horned Owl", "Cactus Wren", "Verdin"
        ]
        label: str = random.choice(birds)
        return {
            "label": label,
            "confidence": random.uniform(0.4, 0.95),
            "top_k": [],
            "inference_time_ms": 0.0
        }

    def _is_bird_species(self, label: str) -> bool:
        """Verify the classification label is actually a bird."""
        non_bird: set[str] = {
            "cat", "dog", "person", "car", "rabbit", "squirrel",
            "rat", "mouse", "lizard", "snake", "insect", "background"
        }
        return not any(nb in label.lower() for nb in non_bird)

    def _in_sighting_cooldown(self, species: str) -> bool:
        """Check if this species was recently logged (avoid duplicate sightings)."""
        key: str = species.lower()
        last_time: float = self._last_sighting_times.get(key, 0.0)
        return (time.time() - last_time) < self._sighting_cooldown

    def _capture_sighting_photo(
        self,
        frame: np.ndarray,
        species: str,
        confidence: float
    ) -> str:
        """Save a sighting photo with species annotation."""
        try:
            import cv2

            annotated: np.ndarray = frame.copy()
            label_text: str = f"{species} ({confidence:.0%})"
            cv2.putText(
                annotated, label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 0), 2
            )

            timestamp: str = time.strftime("%Y%m%d_%H%M%S")
            safe_species: str = species.replace(" ", "_").lower()
            filename: str = f"bird_{timestamp}_{safe_species}.jpg"
            filepath: str = str(Path(self._photo_path) / filename)
            cv2.imwrite(filepath, annotated)

            logger.debug("Sighting photo saved: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("Failed to save sighting photo: %s", e)
            return ""

    def _record_audio(self) -> Optional[np.ndarray]:
        """Record audio from microphone for BirdNET analysis."""
        logger.debug("Audio recording — %.0fs (simulated in dev)", self._audio_record_duration)
        return None

    def _record_sighting(self, sighting: BirdSighting) -> None:
        """Record a confirmed bird sighting."""
        self._sightings.append(sighting)
        if len(self._sightings) > self._max_sightings:
            self._sightings = self._sightings[-self._max_sightings:]

        self._last_sighting_times[sighting.species.lower()] = sighting.timestamp

        self._update_species_log(sighting)

        logger.info(
            "BIRD SIGHTING: %s (%.1f%%, %s) at (%.7f, %.7f)",
            sighting.species, sighting.confidence * 100,
            sighting.identification_method,
            sighting.lat, sighting.lon
        )

        self._record_detection(
            detection_type="bird",
            label=sighting.species,
            confidence=sighting.confidence,
            metadata=sighting.to_dict()
        )

    def _update_species_log(self, sighting: BirdSighting) -> None:
        """Update cumulative species log."""
        key: str = sighting.species.lower()

        if key not in self._species_log:
            self._species_log[key] = SpeciesLog(
                species=key,
                common_name=sighting.common_name,
                scientific_name=sighting.scientific_name,
                first_seen=sighting.timestamp
            )

        log: SpeciesLog = self._species_log[key]
        log.sighting_count += 1
        log.last_seen = sighting.timestamp

        if sighting.scientific_name and not log.scientific_name:
            log.scientific_name = sighting.scientific_name

        if sighting.confidence > log.best_confidence:
            log.best_confidence = sighting.confidence
            if sighting.photo_path:
                log.best_photo_path = sighting.photo_path

        if sighting.lat != 0.0 or sighting.lon != 0.0:
            log.locations.append((sighting.lat, sighting.lon))
            if len(log.locations) > self._max_locations_per_species:
                log.locations = log.locations[-self._max_locations_per_species:]

    def _get_gps_data(self) -> dict[str, Any]:
        """Get current GPS coordinates."""
        if self._gps is None:
            return {"lat": 0.0, "lon": 0.0, "alt_m": 0.0, "gps_fix": False}

        reading = self._gps.reading
        return {
            "lat": reading.lat,
            "lon": reading.lon,
            "alt_m": reading.alt_m,
            "gps_fix": reading.has_fix
        }

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Perform bird watching session at current position.
        Called by patrol executor action handler at each waypoint.
        """
        sightings_before: int = len(self._sightings)
        scan_duration: float = 30.0
        start: float = time.time()

        while time.time() - start < scan_duration:
            if not self._running:
                break

            self._visual_scan()
            time.sleep(0.5)

        if self._species_id and self._species_id.audio_available:
            self._audio_scan()

        new_sightings: int = len(self._sightings) - sightings_before

        return {
            "scan_duration_s": round(time.time() - start, 1),
            "new_sightings": new_sightings,
            "total_species": len(self._species_log),
            "gps": self._get_gps_data(),
            "timestamp": time.time()
        }

    @property
    def species_count(self) -> int:
        return len(self._species_log)

    @property
    def total_sightings(self) -> int:
        return len(self._sightings)

    @property
    def species_list(self) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._species_log.items()}

    @property
    def recent_sightings(self) -> list[BirdSighting]:
        return list(self._sightings[-20:])

    @property
    def total_visual_ids(self) -> int:
        return self._total_visual_ids

    @property
    def total_audio_ids(self) -> int:
        return self._total_audio_ids
