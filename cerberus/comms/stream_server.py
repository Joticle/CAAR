"""
Cerberus MJPEG Streaming Server
Flask-based HTTP server that serves live camera frames as an
MJPEG stream. Accessible from the Dashboard or any browser.
Runs in a background thread so it doesn't block the brain.
"""

import time
import logging
import threading
from typing import Any, Optional, Generator

from flask import Flask, Response, render_template_string

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)

STREAM_PAGE: str = """
<!DOCTYPE html>
<html>
<head><title>Cerberus Live Feed</title></head>
<body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh;">
<img src="/stream" style="max-width:100%;max-height:100vh;">
</body>
</html>
"""


class StreamServer:
    """
    MJPEG streaming server for Cerberus.
    Grabs JPEG frames from the camera interface and serves them
    over HTTP as a multipart MJPEG stream. Any browser or MJPEG
    client can connect to view the live feed.

    Endpoints:
        /       — HTML page with embedded stream
        /stream — Raw MJPEG stream
        /health — Server health check (JSON)
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._host: str = config.get("streaming", "host", default="0.0.0.0")
        self._port: int = config.get("streaming", "port", default=8080)
        self._framerate: int = config.get("streaming", "framerate", default=15)
        self._jpeg_quality: int = config.get("streaming", "jpeg_quality", default=70)
        self._enabled: bool = config.get("streaming", "enabled", default=True)

        self._camera: Optional[Any] = None
        self._app: Flask = Flask(__name__)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._client_count: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._frame_interval: float = 1.0 / max(1, self._framerate)

        self._setup_routes()

        logger.info(
            "Stream server created — %s:%d, %dfps, quality=%d",
            self._host, self._port, self._framerate, self._jpeg_quality
        )

    def bind_camera(self, camera: Any) -> None:
        """Bind camera interface for frame capture."""
        self._camera = camera

    def _setup_routes(self) -> None:
        """Register Flask routes."""

        @self._app.route("/")
        def index() -> str:
            return render_template_string(STREAM_PAGE)

        @self._app.route("/stream")
        def stream() -> Response:
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self._app.route("/health")
        def health() -> dict[str, Any]:
            return {
                "status": "running" if self._running else "stopped",
                "clients": self._client_count,
                "camera_bound": self._camera is not None,
                "framerate": self._framerate,
                "port": self._port
            }

    def _generate_frames(self) -> Generator[bytes, None, None]:
        """Generate MJPEG frames for streaming."""
        with self._lock:
            self._client_count += 1

        client_id: int = self._client_count
        logger.info("Stream client %d connected", client_id)

        try:
            while self._running:
                jpeg: Optional[bytes] = self._get_frame()

                if jpeg is None:
                    time.sleep(self._frame_interval)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )

                time.sleep(self._frame_interval)

        except GeneratorExit:
            pass
        except Exception as e:
            logger.error("Stream error for client %d: %s", client_id, e)
        finally:
            with self._lock:
                self._client_count = max(0, self._client_count - 1)
            logger.info("Stream client %d disconnected", client_id)

    def _get_frame(self) -> Optional[bytes]:
        """Get a JPEG frame from the camera."""
        if self._camera is None:
            return self._placeholder_frame()

        try:
            return self._camera.capture_jpeg(quality=self._jpeg_quality)
        except Exception as e:
            logger.error("Frame capture error: %s", e)
            return None

    def _placeholder_frame(self) -> bytes:
        """Generate a placeholder frame when no camera is bound."""
        try:
            import cv2
            import numpy as np

            frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame, "CERBERUS",
                (200, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 0), 3
            )
            cv2.putText(
                frame, "No Camera Feed",
                (210, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 150, 0), 2
            )

            _, buffer = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            return buffer.tobytes()

        except Exception:
            return b""

    def start(self) -> bool:
        """Start the streaming server in a background thread."""
        if not self._enabled:
            logger.info("Stream server disabled in config")
            return False

        if self._running:
            logger.warning("Stream server already running")
            return False

        self._running = True

        self._thread = threading.Thread(
            target=self._run_server,
            name="stream-server",
            daemon=True
        )
        self._thread.start()

        logger.info("Stream server started on http://%s:%d", self._host, self._port)
        return True

    def _run_server(self) -> None:
        """Run the Flask server (blocking call in background thread)."""
        log: logging.Logger = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)

        try:
            self._app.run(
                host=self._host,
                port=self._port,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            if self._running:
                logger.error("Stream server error: %s", e)

    def stop(self) -> None:
        """Stop the streaming server."""
        if not self._running:
            return

        self._running = False
        logger.info("Stream server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        with self._lock:
            return self._client_count

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def __repr__(self) -> str:
        return (
            f"StreamServer(running={self._running}, "
            f"clients={self._client_count}, "
            f"url={self.url})"
        )