"""
Cerberus Motion Detection Pipeline
OpenCV-based motion detection for the surveillance head.
Detects movement by comparing consecutive frames using background
subtraction. Filters noise, calculates motion regions, and triggers
callbacks when significant motion is detected.
All processing at the edge — no cloud dependency.
"""

import time
import logging
import threading
from typing import Any, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
import cv2

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MotionEvent:
    """Represents a detected motion event."""
    detected: bool = False
    region_count: int = 0
    total_area: int = 0
    largest_area: int = 0
    motion_pct: float = 0.0
    bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    center_points: list[tuple[int, int]] = field(default_factory=list)
    frame_index: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "region_count": self.region_count,
            "total_area": self.total_area,
            "largest_area": self.largest_area,
            "motion_pct": round(self.motion_pct, 2),
            "bounding_boxes": self.bounding_boxes,
            "center_points": self.center_points,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp
        }


class MotionDetector:
    """
    OpenCV motion detection pipeline.
    Uses MOG2 background subtractor with morphological filtering
    to detect significant movement in camera frames.

    Pipeline:
        1. Convert frame to grayscale
        2. Apply Gaussian blur to reduce noise
        3. Feed to background subtractor (MOG2)
        4. Threshold the foreground mask
        5. Apply morphological operations (dilate/erode) to clean up
        6. Find contours in the cleaned mask
        7. Filter contours by minimum area
        8. Generate MotionEvent with bounding boxes

    Consumers: surveillance head, pest deterrent head.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._min_area: int = config.get("intelligence", "motion", "min_area", default=500)
        self._threshold: int = config.get("intelligence", "motion", "threshold", default=25)
        self._blur_size: int = config.get("intelligence", "motion", "blur_size", default=21)
        self._history: int = config.get("intelligence", "motion", "history", default=500)
        self._var_threshold: float = config.get("intelligence", "motion", "var_threshold", default=40.0)
        self._detect_shadows: bool = config.get("intelligence", "motion", "detect_shadows", default=True)
        self._dilate_iterations: int = config.get("intelligence", "motion", "dilate_iterations", default=3)
        self._erode_iterations: int = config.get("intelligence", "motion", "erode_iterations", default=1)
        self._cooldown_seconds: float = config.get("intelligence", "motion", "cooldown_seconds", default=2.0)
        self._max_regions: int = config.get("intelligence", "motion", "max_regions", default=20)
        self._motion_pct_threshold: float = config.get(
            "intelligence", "motion", "motion_pct_threshold", default=0.5
        )

        self._bg_subtractor: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=self._detect_shadows
        )

        self._kernel_dilate: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._kernel_erode: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._frame_count: int = 0
        self._last_motion_time: float = 0.0
        self._enabled: bool = True
        self._warmup_frames: int = 30
        self._lock: threading.Lock = threading.Lock()

        self._motion_callbacks: list[Callable[[MotionEvent], None]] = []
        self._motion_start_callbacks: list[Callable[[], None]] = []
        self._motion_stop_callbacks: list[Callable[[], None]] = []
        self._in_motion: bool = False

        logger.info(
            "Motion detector created — min_area=%d, threshold=%d, blur=%d, cooldown=%.1fs",
            self._min_area, self._threshold, self._blur_size, self._cooldown_seconds
        )

    def process_frame(self, frame: np.ndarray) -> MotionEvent:
        """
        Process a single frame through the motion detection pipeline.
        frame: numpy array (H, W, 3) in BGR format from camera.
        Returns MotionEvent with detection results.
        """
        if not self._enabled:
            return MotionEvent(timestamp=time.time())

        with self._lock:
            self._frame_count += 1
            frame_idx: int = self._frame_count

        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred: np.ndarray = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)

        fg_mask: np.ndarray = self._bg_subtractor.apply(blurred)

        _, thresh: np.ndarray = cv2.threshold(fg_mask, self._threshold, 255, cv2.THRESH_BINARY)

        dilated: np.ndarray = cv2.dilate(
            thresh, self._kernel_dilate, iterations=self._dilate_iterations
        )
        cleaned: np.ndarray = cv2.erode(
            dilated, self._kernel_erode, iterations=self._erode_iterations
        )

        if frame_idx <= self._warmup_frames:
            return MotionEvent(frame_index=frame_idx, timestamp=time.time())

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        event: MotionEvent = self._analyze_contours(contours, frame.shape, frame_idx)

        self._handle_motion_state(event)

        return event

    def _analyze_contours(
        self,
        contours: list,
        frame_shape: tuple,
        frame_idx: int
    ) -> MotionEvent:
        """Analyze contours to build a MotionEvent."""
        bounding_boxes: list[tuple[int, int, int, int]] = []
        center_points: list[tuple[int, int]] = []
        total_area: int = 0
        largest_area: int = 0

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours[:self._max_regions]:
            area: int = int(cv2.contourArea(contour))
            if area < self._min_area:
                continue

            x: int
            y: int
            w: int
            h: int
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

            cx: int = x + w // 2
            cy: int = y + h // 2
            center_points.append((cx, cy))

            total_area += area
            if area > largest_area:
                largest_area = area

        frame_area: int = frame_shape[0] * frame_shape[1]
        motion_pct: float = (total_area / frame_area * 100) if frame_area > 0 else 0.0
        detected: bool = len(bounding_boxes) > 0 and motion_pct >= self._motion_pct_threshold

        return MotionEvent(
            detected=detected,
            region_count=len(bounding_boxes),
            total_area=total_area,
            largest_area=largest_area,
            motion_pct=motion_pct,
            bounding_boxes=bounding_boxes,
            center_points=center_points,
            frame_index=frame_idx,
            timestamp=time.time()
        )

    def _handle_motion_state(self, event: MotionEvent) -> None:
        """Track motion state transitions and fire callbacks."""
        now: float = time.time()

        if event.detected:
            self._last_motion_time = now

            if not self._in_motion:
                self._in_motion = True
                logger.info(
                    "Motion STARTED — %d regions, %.1f%% of frame, largest=%dpx",
                    event.region_count, event.motion_pct, event.largest_area
                )
                for cb in self._motion_start_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        logger.error("Motion start callback error: %s", e)

            for cb in self._motion_callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.error("Motion callback error: %s", e)

        elif self._in_motion:
            elapsed: float = now - self._last_motion_time
            if elapsed >= self._cooldown_seconds:
                self._in_motion = False
                logger.info("Motion STOPPED — cooldown elapsed (%.1fs)", elapsed)
                for cb in self._motion_stop_callbacks:
                    try:
                        cb()
                    except Exception as e:
                        logger.error("Motion stop callback error: %s", e)

    def annotate_frame(
        self,
        frame: np.ndarray,
        event: MotionEvent,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame copy.
        Returns annotated frame — does not modify the original.
        """
        annotated: np.ndarray = frame.copy()

        if not event.detected:
            return annotated

        for i, (x, y, w, h) in enumerate(event.bounding_boxes):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

            label: str = f"Motion {i + 1}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x, y - label_size[1] - 4),
                (x + label_size[0], y),
                color, -1
            )
            cv2.putText(
                annotated, label,
                (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1
            )

        status: str = f"Regions: {event.region_count} | Area: {event.motion_pct:.1f}%"
        cv2.putText(
            annotated, status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2
        )

        return annotated

    def register_motion_callback(self, callback: Callable[[MotionEvent], None]) -> None:
        """Register callback for each motion event: callback(event)."""
        self._motion_callbacks.append(callback)

    def register_motion_start_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for motion start: callback()."""
        self._motion_start_callbacks.append(callback)

    def register_motion_stop_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for motion stop: callback()."""
        self._motion_stop_callbacks.append(callback)

    def reset(self) -> None:
        """Reset the background model. Use after major scene changes."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=self._detect_shadows
        )
        with self._lock:
            self._frame_count = 0
        self._in_motion = False
        self._last_motion_time = 0.0
        logger.info("Motion detector reset — background model cleared")

    def enable(self) -> None:
        """Enable motion detection."""
        self._enabled = True
        logger.info("Motion detection enabled")

    def disable(self) -> None:
        """Disable motion detection."""
        self._enabled = False
        self._in_motion = False
        logger.info("Motion detection disabled")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_in_motion(self) -> bool:
        return self._in_motion

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def warmed_up(self) -> bool:
        with self._lock:
            return self._frame_count > self._warmup_frames

    def __repr__(self) -> str:
        return (
            f"MotionDetector(enabled={self._enabled}, "
            f"in_motion={self._in_motion}, "
            f"frames={self._frame_count}, "
            f"warmed_up={self.warmed_up})"
        )