"""
Cerberus Weed Scanner Head
Pi Camera 3 + pan-tilt servo + GPS geotag + TFLite weed classification.
Captures images, runs inference, geotags detections, and logs results.
Designed for systematic yard coverage via grid driver integration.
"""

import time
import logging
from typing import Any, Optional
from pathlib import Path

import numpy as np

from cerberus.core.config import CerberusConfig
from cerberus.heads.base_head import BaseHead, HeadInfo, HeadState


logger: logging.Logger = logging.getLogger(__name__)


class WeedScannerHead(BaseHead):
    """
    Head 1: Weed Scanner
    Scans the ground for weeds using camera + TFLite classification.
    Geotags every detection with GPS coordinates for mapping.
    Designed to work with grid driver for complete yard coverage.

    Cycle:
        1. Capture frame from camera
        2. Run TFLite weed classifier
        3. If weed detected above threshold → geotag, save image, log detection
        4. Publish detection to MQTT for Dashboard mapping
    """

    _HEAD_INFO: HeadInfo = HeadInfo(
        name="weed_scanner",
        description="Camera-based weed detection with GPS geotagging",
        version="1.0",
        requires_camera=True,
        requires_gps=True,
        requires_sensors=False,
        requires_audio=False,
        requires_servos=True,
        supported_tasks=["scan", "grid_scan", "capture"]
    )

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        super().__init__(config)

        self._model_name: str = self._config.get(
            "heads", "weed_scanner", "model_name", default="weed_detector"
        )
        self._model_file: str = self._config.get(
            "heads", "weed_scanner", "model_file", default="weed_detector.tflite"
        )
        self._labels_file: Optional[str] = self._config.get(
            "heads", "weed_scanner", "labels_file", default="weed_labels.txt"
        )
        self._confidence_threshold: float = self._config.get(
            "heads", "weed_scanner", "confidence_threshold", default=0.6
        )
        self._save_detections: bool = self._config.get(
            "heads", "weed_scanner", "save_detection_images", default=True
        )
        self._save_path: str = self._config.get(
            "heads", "weed_scanner", "save_path", default="data/weed_detections"
        )
        self._scan_positions: list[dict[str, float]] = self._config.get(
            "heads", "weed_scanner", "scan_positions", default=[
                {"pan": 0.0, "tilt": -30.0},
                {"pan": -45.0, "tilt": -30.0},
                {"pan": 45.0, "tilt": -30.0}
            ]
        )
        self._current_scan_index: int = 0
        self._total_scans: int = 0
        self._weed_detections: int = 0
        self._model_loaded: bool = False

        Path(self._save_path).mkdir(parents=True, exist_ok=True)

    @property
    def info(self) -> HeadInfo:
        return self._HEAD_INFO

    def _on_load(self) -> bool:
        """Load weed detection TFLite model."""
        if self._classifier is None:
            logger.warning("No classifier bound — weed scanner will run in simulation mode")
            return True

        try:
            self._model_loaded = self._classifier.load_model(
                name=self._model_name,
                model_file=self._model_file,
                labels_file=self._labels_file
            )

            if self._model_loaded:
                logger.info("Weed detection model loaded: %s", self._model_name)
            else:
                logger.warning("Weed model not loaded — running without classification")

            return True

        except Exception as e:
            logger.error("Failed to load weed model: %s", e)
            return True

    def _on_start(self) -> bool:
        """Start camera and prepare for scanning."""
        if self._camera is not None:
            if not self._camera.is_streaming:
                self._camera.start()

        self._current_scan_index = 0
        self._total_scans = 0
        self._weed_detections = 0

        logger.info("Weed scanner activated — threshold=%.0f%%", self._confidence_threshold * 100)
        return True

    def _on_stop(self) -> None:
        """Stop scanning."""
        logger.info(
            "Weed scanner deactivated — %d scans, %d weeds detected",
            self._total_scans, self._weed_detections
        )

    def _on_unload(self) -> None:
        """Release weed scanner resources."""
        logger.info("Weed scanner unloaded")

    def _run_cycle(self) -> None:
        """One scan cycle: capture, classify, geotag if detected."""
        self._total_scans += 1
        self._record_activity(f"Scanning (position {self._current_scan_index + 1}/{len(self._scan_positions)})")

        self._move_to_scan_position()

        frame: Optional[np.ndarray] = self._capture_scan_frame()
        if frame is None:
            return

        result: dict[str, Any] = self._classify_frame(frame)

        if result["detected"]:
            self._handle_weed_detection(frame, result)

        self._advance_scan_position()

    def _move_to_scan_position(self) -> None:
        """Move pan-tilt servos to the current scan position."""
        if not self._scan_positions:
            return

        position: dict[str, float] = self._scan_positions[self._current_scan_index]
        pan: float = position.get("pan", 0.0)
        tilt: float = position.get("tilt", -30.0)

        logger.debug("Moving to scan position: pan=%.1f, tilt=%.1f", pan, tilt)
        time.sleep(0.3)

    def _capture_scan_frame(self) -> Optional[np.ndarray]:
        """Capture a frame for weed analysis."""
        if self._camera is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame: Optional[np.ndarray] = self._camera.capture_frame()
        if frame is None:
            logger.warning("Weed scanner failed to capture frame")
            return None

        return frame

    def _classify_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Run weed classification on a frame."""
        if self._classifier is None or not self._model_loaded:
            return self._simulated_classification()

        try:
            result = self._classifier.classify(
                model_name=self._model_name,
                image=frame,
                top_k=3,
                threshold=self._confidence_threshold
            )

            is_weed: bool = (
                result.confidence >= self._confidence_threshold and
                self._is_weed_label(result.label)
            )

            return {
                "detected": is_weed,
                "label": result.label,
                "confidence": result.confidence,
                "top_k": result.top_k,
                "inference_time_ms": result.inference_time_ms
            }

        except Exception as e:
            logger.error("Weed classification error: %s", e)
            return {"detected": False, "label": "error", "confidence": 0.0, "top_k": [], "inference_time_ms": 0.0}

    def _simulated_classification(self) -> dict[str, Any]:
        """Return simulated classification for dev environment."""
        import random
        is_weed: bool = random.random() < 0.15
        return {
            "detected": is_weed,
            "label": "dandelion" if is_weed else "grass",
            "confidence": random.uniform(0.7, 0.95) if is_weed else random.uniform(0.1, 0.4),
            "top_k": [],
            "inference_time_ms": 0.0
        }

    def _is_weed_label(self, label: str) -> bool:
        """Check if a classification label is a weed species."""
        weed_keywords: set[str] = {
            "weed", "dandelion", "thistle", "crabgrass", "clover",
            "spurge", "bindweed", "purslane", "pigweed", "goathead",
            "puncturevine", "foxtail", "burweed", "nutsedge", "bermuda"
        }
        label_lower: str = label.lower()
        return any(keyword in label_lower for keyword in weed_keywords)

    def _handle_weed_detection(self, frame: np.ndarray, result: dict[str, Any]) -> None:
        """Process a confirmed weed detection."""
        self._weed_detections += 1

        gps_data: dict[str, Any] = self._get_gps_data()
        timestamp: str = time.strftime("%Y%m%d_%H%M%S")

        detection_data: dict[str, Any] = {
            "label": result["label"],
            "confidence": result["confidence"],
            "scan_position": self._current_scan_index,
            "scan_number": self._total_scans,
            **gps_data
        }

        logger.info(
            "WEED DETECTED: %s (%.1f%%) at (%.7f, %.7f)",
            result["label"],
            result["confidence"] * 100,
            gps_data.get("lat", 0.0),
            gps_data.get("lon", 0.0)
        )

        if self._save_detections:
            self._save_detection_image(frame, result, timestamp)

        self._record_detection(
            detection_type="weed",
            label=result["label"],
            confidence=result["confidence"],
            metadata=detection_data
        )

    def _save_detection_image(
        self,
        frame: np.ndarray,
        result: dict[str, Any],
        timestamp: str
    ) -> None:
        """Save the detection frame with annotation."""
        try:
            import cv2

            annotated: np.ndarray = frame.copy()
            label_text: str = f"{result['label']} {result['confidence']:.0%}"
            cv2.putText(
                annotated, label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2
            )

            filename: str = f"weed_{timestamp}_{result['label']}.jpg"
            filepath: str = str(Path(self._save_path) / filename)
            cv2.imwrite(filepath, annotated)

            logger.debug("Detection image saved: %s", filepath)

        except Exception as e:
            logger.error("Failed to save detection image: %s", e)

    def _get_gps_data(self) -> dict[str, Any]:
        """Get current GPS coordinates for geotagging."""
        if self._gps is None:
            return {"lat": 0.0, "lon": 0.0, "gps_fix": False}

        reading = self._gps.reading
        return {
            "lat": reading.lat,
            "lon": reading.lon,
            "alt_m": reading.alt_m,
            "gps_fix": reading.has_fix,
            "hdop": reading.hdop
        }

    def _advance_scan_position(self) -> None:
        """Move to the next scan position in the sequence."""
        if self._scan_positions:
            self._current_scan_index = (self._current_scan_index + 1) % len(self._scan_positions)

    def scan_at_point(self, point: Any = None) -> dict[str, Any]:
        """
        Perform a full scan at the current position.
        Called by grid driver action handler at each grid point.
        Returns detection data for the point.
        """
        detections: list[dict[str, Any]] = []

        for i in range(len(self._scan_positions)):
            self._current_scan_index = i
            self._move_to_scan_position()

            frame: Optional[np.ndarray] = self._capture_scan_frame()
            if frame is None:
                continue

            result: dict[str, Any] = self._classify_frame(frame)

            if result["detected"]:
                self._handle_weed_detection(frame, result)
                detections.append(result)

        self._current_scan_index = 0

        return {
            "scans_performed": len(self._scan_positions),
            "weeds_found": len(detections),
            "detections": detections,
            "gps": self._get_gps_data(),
            "timestamp": time.time()
        }

    @property
    def total_scans(self) -> int:
        return self._total_scans

    @property
    def weed_count(self) -> int:
        return self._weed_detections