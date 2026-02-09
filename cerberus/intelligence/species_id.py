"""
Cerberus Species Identification
Wraps BirdNET for audio-based bird species identification and
TFLite models for visual wildlife classification.
Powers the bird watcher head and pest deterrent head.
All inference at the edge — no cloud dependency.
"""

import time
import logging
import wave
import tempfile
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from cerberus.core.config import CerberusConfig
from cerberus.intelligence.classifier import Classifier, ClassificationResult


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class AudioSpeciesResult:
    """Result of audio-based species identification."""
    species: str = "unknown"
    common_name: str = "unknown"
    scientific_name: str = "unknown"
    confidence: float = 0.0
    top_results: list[dict[str, Any]] = field(default_factory=list)
    audio_duration_s: float = 0.0
    inference_time_ms: float = 0.0
    timestamp: float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "confidence": round(self.confidence, 4),
            "top_results": self.top_results,
            "audio_duration_s": round(self.audio_duration_s, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "timestamp": self.timestamp
        }


@dataclass
class VisualSpeciesResult:
    """Result of visual species identification."""
    species: str = "unknown"
    category: str = "unknown"
    confidence: float = 0.0
    is_pest: bool = False
    is_bird: bool = False
    classification: ClassificationResult = field(default_factory=ClassificationResult)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "is_pest": self.is_pest,
            "is_bird": self.is_bird,
            "classification": self.classification.to_dict(),
            "timestamp": self.timestamp
        }


class BirdNETAnalyzer:
    """
    Wrapper around the BirdNET-Analyzer library for audio bird ID.
    BirdNET identifies bird species from audio recordings using a
    pre-trained neural network. Runs entirely on-device.

    Requires: birdnetlib (pip install birdnetlib)
    """

    def __init__(self, config: CerberusConfig) -> None:
        self._lat: float = config.get("navigation", "home_lat", default=36.1699)
        self._lon: float = config.get("navigation", "home_lon", default=-115.1398)
        self._min_confidence: float = config.get(
            "intelligence", "species", "audio_min_confidence", default=0.25
        )
        self._sample_rate: int = config.get(
            "intelligence", "species", "sample_rate", default=48000
        )
        self._analyzer: Optional[Any] = None
        self._available: bool = False

        self._init_birdnet()

    def _init_birdnet(self) -> None:
        """Initialize BirdNET analyzer."""
        try:
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer

            self._analyzer = Analyzer()
            self._available = True
            logger.info("BirdNET analyzer initialized")

        except ImportError:
            logger.warning("birdnetlib not available — audio species ID disabled (dev mode)")
            self._available = False

        except Exception as e:
            logger.error("BirdNET initialization failed: %s", e)
            self._available = False

    def identify_file(self, audio_path: str) -> AudioSpeciesResult:
        """
        Identify bird species from an audio file.
        Supports WAV format. Returns best match with top results.
        """
        if not self._available or self._analyzer is None:
            return self._simulated_result()

        if not Path(audio_path).exists():
            logger.error("Audio file not found: %s", audio_path)
            return AudioSpeciesResult(timestamp=time.time())

        try:
            from birdnetlib import Recording

            start_time: float = time.time()

            recording: Recording = Recording(
                self._analyzer,
                audio_path,
                lat=self._lat,
                lon=self._lon,
                min_conf=self._min_confidence
            )
            recording.analyze()

            elapsed_ms: float = (time.time() - start_time) * 1000

            return self._parse_birdnet_results(recording.detections, elapsed_ms, audio_path)

        except Exception as e:
            logger.error("BirdNET analysis failed for %s: %s", audio_path, e)
            return AudioSpeciesResult(timestamp=time.time())

    def identify_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> AudioSpeciesResult:
        """
        Identify bird species from raw audio data.
        audio_data: numpy array of audio samples.
        Writes to a temp WAV file for BirdNET processing.
        """
        if not self._available:
            return self._simulated_result()

        if sample_rate is None:
            sample_rate = self._sample_rate

        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name

            self._write_wav(temp_path, audio_data, sample_rate)
            result: AudioSpeciesResult = self.identify_file(temp_path)
            return result

        except Exception as e:
            logger.error("Audio data identification failed: %s", e)
            return AudioSpeciesResult(timestamp=time.time())

        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _write_wav(self, path: str, data: np.ndarray, sample_rate: int) -> None:
        """Write numpy audio data to a WAV file."""
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (data * 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)

        channels: int = 1 if data.ndim == 1 else data.shape[1]

        with wave.open(path, "w") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())

    def _parse_birdnet_results(
        self,
        detections: list[dict],
        inference_time_ms: float,
        audio_path: str
    ) -> AudioSpeciesResult:
        """Parse BirdNET detection results."""
        if not detections:
            return AudioSpeciesResult(
                inference_time_ms=inference_time_ms,
                timestamp=time.time()
            )

        sorted_detections: list[dict] = sorted(
            detections, key=lambda d: d.get("confidence", 0), reverse=True
        )

        top_results: list[dict[str, Any]] = []
        for det in sorted_detections[:5]:
            common: str = det.get("common_name", "unknown")
            scientific: str = det.get("scientific_name", "unknown")
            conf: float = det.get("confidence", 0.0)
            top_results.append({
                "common_name": common,
                "scientific_name": scientific,
                "confidence": round(conf, 4),
                "start_time": det.get("start_time", 0),
                "end_time": det.get("end_time", 0)
            })

        best: dict = sorted_detections[0]
        common_name: str = best.get("common_name", "unknown")
        scientific_name: str = best.get("scientific_name", "unknown")

        try:
            with wave.open(audio_path, "r") as wf:
                duration: float = wf.getnframes() / float(wf.getframerate())
        except Exception:
            duration = 0.0

        return AudioSpeciesResult(
            species=common_name,
            common_name=common_name,
            scientific_name=scientific_name,
            confidence=best.get("confidence", 0.0),
            top_results=top_results,
            audio_duration_s=duration,
            inference_time_ms=inference_time_ms,
            timestamp=time.time()
        )

    def _simulated_result(self) -> AudioSpeciesResult:
        """Return a simulated result for dev environment."""
        return AudioSpeciesResult(
            species="Simulated Bird",
            common_name="Simulated Bird",
            scientific_name="Avis simulatus",
            confidence=0.0,
            top_results=[{
                "common_name": "Simulated Bird",
                "scientific_name": "Avis simulatus",
                "confidence": 0.0,
                "start_time": 0,
                "end_time": 3
            }],
            audio_duration_s=3.0,
            inference_time_ms=0.0,
            timestamp=time.time()
        )

    @property
    def available(self) -> bool:
        return self._available


class VisualSpeciesClassifier:
    """
    Visual species classifier using TFLite models.
    Identifies wildlife from camera images — birds, pests, general wildlife.
    Uses the Classifier manager to load and run appropriate models.
    """

    PEST_SPECIES: set[str] = {
        "rabbit", "squirrel", "rat", "mouse", "gopher",
        "pigeon", "starling", "grackle", "crow", "coyote",
        "snake", "lizard", "scorpion", "cockroach", "cricket"
    }

    BIRD_SPECIES: set[str] = {
        "sparrow", "finch", "dove", "pigeon", "hawk", "falcon",
        "owl", "hummingbird", "woodpecker", "robin", "cardinal",
        "jay", "mockingbird", "starling", "grackle", "crow",
        "raven", "quail", "roadrunner", "vulture", "eagle",
        "wren", "warbler", "thrush", "oriole", "tanager"
    }

    def __init__(
        self,
        classifier: Classifier,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._classifier: Classifier = classifier
        self._visual_model: str = config.get(
            "intelligence", "species", "visual_model", default="wildlife_classifier"
        )
        self._visual_model_file: str = config.get(
            "intelligence", "species", "visual_model_file", default="wildlife_classifier.tflite"
        )
        self._visual_labels_file: Optional[str] = config.get(
            "intelligence", "species", "visual_labels_file", default="wildlife_labels.txt"
        )
        self._min_confidence: float = config.get(
            "intelligence", "species", "visual_min_confidence", default=0.4
        )

        self._load_model()

        logger.info("Visual species classifier created — model=%s", self._visual_model)

    def _load_model(self) -> None:
        """Load the visual species classification model."""
        self._classifier.load_model(
            name=self._visual_model,
            model_file=self._visual_model_file,
            labels_file=self._visual_labels_file
        )

    def identify(self, image: np.ndarray) -> VisualSpeciesResult:
        """
        Identify species in an image.
        image: numpy array (H, W, 3) BGR format from camera.
        Returns VisualSpeciesResult with species info and category.
        """
        classification: ClassificationResult = self._classifier.classify(
            model_name=self._visual_model,
            image=image,
            top_k=5,
            threshold=self._min_confidence
        )

        species: str = classification.label.lower()
        category: str = self._categorize(species)
        is_pest: bool = self._is_pest(species)
        is_bird: bool = self._is_bird(species)

        result: VisualSpeciesResult = VisualSpeciesResult(
            species=classification.label,
            category=category,
            confidence=classification.confidence,
            is_pest=is_pest,
            is_bird=is_bird,
            classification=classification,
            timestamp=time.time()
        )

        if classification.confidence >= self._min_confidence:
            logger.info(
                "Species identified: %s (%s) — confidence=%.1f%%, pest=%s, bird=%s",
                classification.label, category,
                classification.confidence * 100,
                is_pest, is_bird
            )

        return result

    def _categorize(self, species: str) -> str:
        """Categorize a species string into a broad group."""
        species_lower: str = species.lower()

        if self._is_bird(species_lower):
            return "bird"
        if self._is_pest(species_lower):
            return "pest"
        if any(w in species_lower for w in ["cat", "dog", "human", "person"]):
            return "domestic"
        return "wildlife"

    def _is_pest(self, species: str) -> bool:
        """Check if a species is in the pest list."""
        species_lower: str = species.lower()
        return any(pest in species_lower for pest in self.PEST_SPECIES)

    def _is_bird(self, species: str) -> bool:
        """Check if a species is in the bird list."""
        species_lower: str = species.lower()
        return any(bird in species_lower for bird in self.BIRD_SPECIES)

    @property
    def model_loaded(self) -> bool:
        model = self._classifier.get_model(self._visual_model)
        return model is not None and model.loaded


class SpeciesIdentifier:
    """
    Unified species identification interface for Cerberus.
    Combines audio (BirdNET) and visual (TFLite) identification.
    Used by bird watcher head and pest deterrent head.
    """

    def __init__(
        self,
        classifier: Classifier,
        config: Optional[CerberusConfig] = None
    ) -> None:
        if config is None:
            config = CerberusConfig()

        self._config: CerberusConfig = config
        self._audio: BirdNETAnalyzer = BirdNETAnalyzer(config)
        self._visual: VisualSpeciesClassifier = VisualSpeciesClassifier(classifier, config)

        self._detection_log: list[dict[str, Any]] = []
        self._max_log_size: int = 1000
        self._lock: threading.Lock = threading.Lock()
        self._db: Optional[Any] = None

        logger.info(
            "Species identifier created — audio=%s, visual=%s",
            "enabled" if self._audio.available else "disabled",
            "enabled" if self._visual.model_loaded else "disabled"
        )

    def bind_db(self, db: Any) -> None:
        """Bind database for detection logging."""
        self._db = db

    def identify_audio(self, audio_path: str) -> AudioSpeciesResult:
        """Identify bird species from audio file."""
        result: AudioSpeciesResult = self._audio.identify_file(audio_path)
        self._log_detection("audio", result.to_dict())
        return result

    def identify_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> AudioSpeciesResult:
        """Identify bird species from raw audio data."""
        result: AudioSpeciesResult = self._audio.identify_audio_data(audio_data, sample_rate)
        self._log_detection("audio", result.to_dict())
        return result

    def identify_visual(self, image: np.ndarray) -> VisualSpeciesResult:
        """Identify species from camera image."""
        result: VisualSpeciesResult = self._visual.identify(image)
        self._log_detection("visual", result.to_dict())
        return result

    def _log_detection(self, method: str, data: dict[str, Any]) -> None:
        """Log a species detection to memory and database."""
        entry: dict[str, Any] = {
            "method": method,
            "data": data,
            "timestamp": time.time()
        }

        with self._lock:
            self._detection_log.append(entry)
            if len(self._detection_log) > self._max_log_size:
                self._detection_log = self._detection_log[-self._max_log_size:]

        if self._db is not None:
            try:
                species: str = data.get("species", data.get("common_name", "unknown"))
                confidence: float = data.get("confidence", 0.0)
                self._db.log_detection(
                    detection_type=f"species_{method}",
                    label=species,
                    confidence=confidence,
                    metadata=str(data)
                )
            except Exception as e:
                logger.error("Failed to log species detection: %s", e)

    @property
    def audio_available(self) -> bool:
        return self._audio.available

    @property
    def visual_available(self) -> bool:
        return self._visual.model_loaded

    @property
    def recent_detections(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._detection_log[-20:])

    @property
    def audio_analyzer(self) -> BirdNETAnalyzer:
        return self._audio

    @property
    def visual_classifier(self) -> VisualSpeciesClassifier:
        return self._visual

    def __repr__(self) -> str:
        return (
            f"SpeciesIdentifier(audio={'yes' if self._audio.available else 'no'}, "
            f"visual={'yes' if self._visual.model_loaded else 'no'}, "
            f"detections={len(self._detection_log)})"
        )
    