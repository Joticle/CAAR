"""
Cerberus Centralized Logger
Configures Python logging with rotating file handlers and console output.
Every subsystem gets a named logger that flows through this configuration.
MQTT log publishing is available when comms are online.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from cerberus.core.config import CerberusConfig


_logger_initialized: bool = False

LOG_FORMAT: str = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


class MQTTLogHandler(logging.Handler):
    """
    Custom log handler that publishes WARNING+ logs to MQTT.
    Lazily binds to the MQTT client to avoid circular imports.
    Only active when MQTT is connected.
    """

    def __init__(self, topic: str = "cerberus/alerts") -> None:
        super().__init__(level=logging.WARNING)
        self._topic: str = topic
        self._mqtt_client: Optional[object] = None

    def bind_mqtt(self, client: object) -> None:
        """Bind an active MQTT client for publishing. Called after MQTT init."""
        self._mqtt_client = client

    def emit(self, record: logging.LogRecord) -> None:
        if self._mqtt_client is None:
            return

        try:
            message: str = self.format(record)
            publish = getattr(self._mqtt_client, "publish", None)
            if callable(publish):
                publish(self._topic, message)
        except Exception:
            self.handleError(record)


_mqtt_handler: Optional[MQTTLogHandler] = None


def setup_logging(config: Optional[CerberusConfig] = None) -> None:
    """
    Initialize the logging system for Cerberus.
    Call once at startup from brain.py. All subsequent logging.getLogger()
    calls throughout the codebase will inherit this configuration.
    """
    global _logger_initialized, _mqtt_handler

    if _logger_initialized:
        return

    if config is None:
        config = CerberusConfig()

    log_level_str: str = config.get("system", "log_level", default="INFO")
    log_dir: str = config.get("system", "log_dir", default="data/logs")
    max_bytes: int = config.get("system", "log_max_bytes", default=10485760)
    backup_count: int = config.get("system", "log_backup_count", default=5)
    alert_topic: str = config.get("mqtt", "topics", "alerts", default="cerberus/alerts")

    log_level: int = getattr(logging, log_level_str.upper(), logging.INFO)

    os.makedirs(log_dir, exist_ok=True)

    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    formatter: logging.Formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file: str = os.path.join(log_dir, "cerberus.log")
    try:
        file_handler: RotatingFileHandler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as e:
        console_handler.handle(logging.LogRecord(
            name="cerberus.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg=f"Failed to create log file {log_file}: {e}",
            args=(),
            exc_info=None
        ))

    error_file: str = os.path.join(log_dir, "cerberus_errors.log")
    try:
        error_handler: RotatingFileHandler = RotatingFileHandler(
            filename=error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    except OSError:
        pass

    _mqtt_handler = MQTTLogHandler(topic=alert_topic)
    _mqtt_handler.setFormatter(formatter)
    root_logger.addHandler(_mqtt_handler)

    _logger_initialized = True

    logger: logging.Logger = logging.getLogger(__name__)
    logger.info("Logging initialized — level=%s, dir=%s", log_level_str, log_dir)


def get_mqtt_log_handler() -> Optional[MQTTLogHandler]:
    """Retrieve the MQTT log handler so comms can bind to it after connecting."""
    return _mqtt_handler


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper. Every module calls this:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)