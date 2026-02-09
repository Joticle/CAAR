"""
Cerberus Camera Interface
Wrapper around picamera2 for the Pi Camera 3.
Handles still capture, video frames for CV processing, and configuration.
Gracefully degrades when camera hardware is not present (dev environment).
"""

import io
import time
import logging
import threading
from typing import Any, Optional
from pathlib import Path

import numpy as np

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


class CameraState:
    """Camera operational states."""
    OFFLINE: str = "offline"
    INITIALIZING: str = "initializing"
    READY: str = "ready"
    STREAMING: str = "streaming"
    ERROR: str = "error"


class CerberusCamera:
    """
    Pi Camera 3 interface for Cerberus.
    Provides still capture, continuous frame capture for CV/ML pipelines,
    and JPEG encoding for MJPEG streaming.

    All capture methods return numpy arrays (OpenCV BGR format) or
    JPEG bytes. Consumers include: surveillance head, weed scanner,
    bird watcher, motion detector, MJPEG stream server.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._still_width: int = config.get("camera", "still_width", default=4608)
        self._still_height: int = config.get("camera", "still_height", default=2592)
        self._stream_width: int = config.get("camera", "stream_width", default=1280)
        self._stream_height: int = config.get("camera", "stream_height", default=720)
        self._inference_width: int = config.get("camera", "inference_width", default=640)
        self._inference_height: int = config.get("camera", "inference_height", default=480)
        self._jpeg_quality: int = config.get("camera", "jpeg_quality", default=80)
        self._framerate: int = config.get("camera", "framerate", default=30)
        self._hflip: bool = config.get("camera", "hflip", default=False)
        self._vflip: bool = config.get("camera", "vflip", default=False)
        self._save_path: str = config.get("camera", "save_path", default="data/captures")

        self._camera: Optional[Any] = None
        self._state: str = CameraState.OFFLINE
        self._hardware_available: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._frame_count: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._streaming: bool = False

        Path(self._save_path).mkdir(parents=True, exist_ok=True)

        self._init_hardware()

        logger.info(
            "Camera created — still=%dx%d, stream=%dx%d, inference=%dx%d, hw=%s",
            self._still_width, self._still_height,
            self._stream_width, self._stream_height,
            self._inference_width, self._inference_height,
            "yes" if self._hardware_available else "sim"
        )

    def _init_hardware(self) -> None:
        """Initialize picamera2. Fails gracefully on dev machines."""
        self._state = CameraState.INITIALIZING
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()

            preview_config = self._camera.create_preview_configuration(
                main={"size": (self._stream_width, self._stream_height), "format": "BGR888"},
                lores={"size": (self._inference_width, self._inference_height), "format": "BGR888"},
                transform=self._get_transform()
            )
            self._camera.configure(preview_config)

            self._hardware_available = True
            self._state = CameraState.READY
            logger.info("Pi Camera 3 initialized")

        except ImportError:
            logger.warning("picamera2 not available — camera running in simulation mode")
            self._hardware_available = False
            self._state = CameraState.READY

        except Exception as e:
            logger.error("Camera initialization failed: %s", e)
            self._hardware_available = False
            self._state = CameraState.ERROR

    def _get_transform(self) -> Any:
        """Create the transform for camera flip settings."""
        try:
            from libcamera import Transform
            return Transform(hflip=self._hflip, vflip=self._vflip)
        except ImportError:
            return None

    def start(self) -> bool:
        """Start the camera for continuous capture."""
        with self._lock:
            if self._state == CameraState.STREAMING:
                logger.warning("Camera already streaming")
                return True

            if not self._hardware_available:
                self._state = CameraState.STREAMING
                self._streaming = True
                self._start_sim_capture()
                logger.info("Camera started (simulation mode)")
                return True

            try:
                self._camera.start()
                time.sleep(0.5)
                self._state = CameraState.STREAMING
                self._streaming = True
                self._start_continuous_capture()
                logger.info("Camera started")
                return True
            except Exception as e:
                logger.error("Failed to start camera: %s", e)
                self._state = CameraState.ERROR
                return False

    def stop(self) -> None:
        """Stop the camera."""
        self._streaming = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None

        with self._lock:
            if self._hardware_available and self._camera is not None:
                try:
                    self._camera.stop()
                except Exception as e:
                    logger.error("Error stopping camera: %s", e)

            self._state = CameraState.READY

        logger.info("Camera stopped")

    def _start_continuous_capture(self) -> None:
        """Start background thread for continuous frame capture."""
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Continuous capture loop — grabs frames for CV consumers."""
        logger.info("Camera capture loop started")
        interval: float = 1.0 / self._framerate

        while self._streaming:
            try:
                frame: np.ndarray = self._camera.capture_array("main")

                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_count += 1

            except Exception as e:
                logger.error("Frame capture error: %s", e)
                time.sleep(0.1)
                continue

            time.sleep(interval)

        logger.info("Camera capture loop stopped — %d frames captured", self._frame_count)

    def _start_sim_capture(self) -> None:
        """Start simulation capture thread — generates test frames."""
        self._capture_thread = threading.Thread(
            target=self._sim_capture_loop,
            name="camera-sim-capture",
            daemon=True
        )
        self._capture_thread.start()

    def _sim_capture_loop(self) -> None:
        """Generate simulated frames for dev environment testing."""
        logger.info("Simulated camera capture started")

        while self._streaming:
            frame: np.ndarray = np.zeros(
                (self._stream_height, self._stream_width, 3),
                dtype=np.uint8
            )

            frame[:, :, 1] = 40
            cx: int = self._stream_width // 2
            cy: int = self._stream_height // 2
            frame[cy - 5:cy + 5, cx - 5:cx + 5] = [0, 255, 0]

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_count += 1

            time.sleep(1.0 / 10)

        logger.info("Simulated camera capture stopped")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame as a numpy array (BGR format).
        Returns None if no frame is available.
        Used by CV/ML pipelines for inference.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def capture_inference_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame resized for ML inference.
        Returns None if no frame is available.
        """
        if not self._hardware_available:
            return np.zeros(
                (self._inference_height, self._inference_width, 3),
                dtype=np.uint8
            )

        with self._lock:
            if self._camera is None:
                return None
            try:
                return self._camera.capture_array("lores")
            except Exception as e:
                logger.error("Inference frame capture error: %s", e)
                return None

    def capture_still(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Capture a full-resolution still image.
        Returns the file path of the saved image, or None on failure.
        """
        if filename is None:
            timestamp: str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

        filepath: str = str(Path(self._save_path) / filename)

        if not self._hardware_available:
            frame: np.ndarray = np.zeros(
                (self._still_height, self._still_width, 3),
                dtype=np.uint8
            )
            try:
                from PIL import Image
                img: Image.Image = Image.fromarray(frame)
                img.save(filepath, quality=self._jpeg_quality)
                logger.info("Simulated still captured: %s", filepath)
                return filepath
            except Exception as e:
                logger.error("Simulated still save error: %s", e)
                return None

        with self._lock:
            if self._camera is None:
                logger.error("Camera not initialized — cannot capture still")
                return None

            try:
                still_config = self._camera.create_still_configuration(
                    main={"size": (self._still_width, self._still_height)},
                    transform=self._get_transform()
                )

                was_streaming: bool = self._streaming
                if was_streaming:
                    self._streaming = False
                    if self._capture_thread is not None:
                        self._capture_thread.join(timeout=2.0)
                    self._camera.stop()

                self._camera.configure(still_config)
                self._camera.start()
                time.sleep(0.5)

                self._camera.capture_file(filepath)

                self._camera.stop()

                if was_streaming:
                    preview_config = self._camera.create_preview_configuration(
                        main={"size": (self._stream_width, self._stream_height), "format": "BGR888"},
                        lores={"size": (self._inference_width, self._inference_height), "format": "BGR888"},
                        transform=self._get_transform()
                    )
                    self._camera.configure(preview_config)
                    self._camera.start()
                    self._streaming = True
                    self._start_continuous_capture()

                logger.info("Still captured: %s", filepath)
                return filepath

            except Exception as e:
                logger.error("Still capture failed: %s", e)
                return None

    def capture_jpeg(self) -> Optional[bytes]:
        """
        Get the latest frame as JPEG bytes.
        Used by MJPEG stream server.
        """
        frame: Optional[np.ndarray] = self.capture_frame()
        if frame is None:
            return None

        try:
            import cv2
            encode_params: list[int] = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success: bool
            buffer: np.ndarray
            success, buffer = cv2.imencode(".jpg", frame, encode_params)
            if success:
                return buffer.tobytes()
            return None
        except Exception as e:
            logger.error("JPEG encoding error: %s", e)
            return None

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def hardware_available(self) -> bool:
        return self._hardware_available

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._stream_width, self._stream_height)

    def release(self) -> None:
        """Release all camera resources. Called during shutdown."""
        self.stop()
        with self._lock:
            if self._hardware_available and self._camera is not None:
                try:
                    self._camera.close()
                except Exception as e:
                    logger.error("Camera release error: %s", e)
            self._camera = None
            self._state = CameraState.OFFLINE
        logger.info("Camera resources released")

    def __repr__(self) -> str:
        return (
            f"CerberusCamera(state='{self._state}', "
            f"frames={self._frame_count}, "
            f"hw={'yes' if self._hardware_available else 'sim'})"
        )