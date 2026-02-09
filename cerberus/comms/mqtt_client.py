"""
Cerberus MQTT Communication Client
Handles all publish/subscribe messaging between Cerberus and the Dashboard.
Auto-reconnects on connection loss. Cerberus operates independently
regardless of MQTT connectivity — comms are a luxury, not a dependency.
"""

import json
import time
import threading
import logging
from typing import Any, Optional, Callable
from dataclasses import dataclass

import paho.mqtt.client as mqtt

from cerberus.core.config import CerberusConfig


logger: logging.Logger = logging.getLogger(__name__)


MQTT_CONNECTED: int = 0
MQTT_DISCONNECTED: int = 1
MQTT_ERROR: int = 2


@dataclass
class MQTTMessage:
    """Parsed inbound MQTT message."""
    topic: str
    payload: Any
    qos: int
    retain: bool
    timestamp: float


class CerberusMQTT:
    """
    MQTT client for Cerberus rover communications.
    Publishes telemetry, detections, alerts. Subscribes to commands.
    Non-blocking, thread-safe, auto-reconnecting.
    Connection loss does not affect rover operation.
    """

    def __init__(self, config: Optional[CerberusConfig] = None) -> None:
        if config is None:
            config = CerberusConfig()

        self._broker_host: str = config.get("mqtt", "broker_host", default="localhost")
        self._broker_port: int = config.get("mqtt", "broker_port", default=1883)
        self._keepalive: int = config.get("mqtt", "keepalive_seconds", default=60)
        self._client_id: str = config.get("mqtt", "client_id", default="cerberus-rover")
        self._reconnect_min: int = config.get("mqtt", "reconnect_min_delay", default=1)
        self._reconnect_max: int = config.get("mqtt", "reconnect_max_delay", default=120)
        self._qos: int = config.get("mqtt", "qos", default=1)
        self._topics: dict[str, str] = config.get("mqtt", "topics", default={})

        self._client: mqtt.Client = mqtt.Client(
            client_id=self._client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(
            min_delay=self._reconnect_min,
            max_delay=self._reconnect_max
        )

        self._state: int = MQTT_DISCONNECTED
        self._lock: threading.Lock = threading.Lock()
        self._command_handlers: dict[str, list[Callable[[MQTTMessage], None]]] = {}
        self._message_queue: list[tuple[str, str, int]] = []
        self._max_queue_size: int = 1000
        self._connected_event: threading.Event = threading.Event()

        logger.info(
            "MQTT client created — broker=%s:%d, client_id=%s",
            self._broker_host, self._broker_port, self._client_id
        )

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None
    ) -> None:
        """Callback when connection to broker is established."""
        if rc == 0:
            with self._lock:
                self._state = MQTT_CONNECTED
            self._connected_event.set()
            logger.info("MQTT connected to %s:%d", self._broker_host, self._broker_port)
            self._subscribe_commands()
            self._flush_queue()
        else:
            with self._lock:
                self._state = MQTT_ERROR
            logger.error("MQTT connection failed with code %d", rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: int = 0,
        properties: Any = None
    ) -> None:
        """Callback when disconnected from broker."""
        with self._lock:
            self._state = MQTT_DISCONNECTED
        self._connected_event.clear()

        if rc == 0:
            logger.info("MQTT disconnected cleanly")
        else:
            logger.warning("MQTT unexpected disconnect (code %d) — will auto-reconnect", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage
    ) -> None:
        """Callback when a subscribed message is received."""
        try:
            try:
                payload: Any = json.loads(msg.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = msg.payload.decode("utf-8", errors="replace")

            message: MQTTMessage = MQTTMessage(
                topic=msg.topic,
                payload=payload,
                qos=msg.qos,
                retain=msg.retain,
                timestamp=time.time()
            )

            logger.debug("MQTT received: %s", msg.topic)
            self._dispatch_message(message)

        except Exception as e:
            logger.error("Error processing MQTT message on %s: %s", msg.topic, e)

    def _dispatch_message(self, message: MQTTMessage) -> None:
        """Route inbound messages to registered command handlers."""
        with self._lock:
            handlers: dict[str, list[Callable]] = dict(self._command_handlers)

        for pattern, callbacks in handlers.items():
            if self._topic_matches(pattern, message.topic):
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(
                            "Command handler error for %s: %s",
                            message.topic, e
                        )

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern with wildcards."""
        if pattern == topic:
            return True

        pattern_parts: list[str] = pattern.split("/")
        topic_parts: list[str] = topic.split("/")

        for i, part in enumerate(pattern_parts):
            if part == "#":
                return True
            if i >= len(topic_parts):
                return False
            if part != "+" and part != topic_parts[i]:
                return False

        return len(pattern_parts) == len(topic_parts)

    def _subscribe_commands(self) -> None:
        """Subscribe to command topics after connection."""
        command_topic: str = self._topics.get("commands", "cerberus/command/#")
        try:
            self._client.subscribe(command_topic, qos=self._qos)
            logger.info("Subscribed to commands: %s", command_topic)
        except Exception as e:
            logger.error("Failed to subscribe to %s: %s", command_topic, e)

    def _flush_queue(self) -> None:
        """Publish any messages that were queued while disconnected."""
        with self._lock:
            queue: list[tuple[str, str, int]] = list(self._message_queue)
            self._message_queue.clear()

        if queue:
            logger.info("Flushing %d queued MQTT messages", len(queue))
            for topic, payload, qos in queue:
                try:
                    self._client.publish(topic, payload, qos=qos)
                except Exception as e:
                    logger.error("Failed to flush queued message to %s: %s", topic, e)

    def connect(self) -> bool:
        """
        Connect to the MQTT broker. Non-blocking.
        Returns True if connection attempt started, False on immediate failure.
        """
        try:
            self._client.connect_async(
                self._broker_host,
                self._broker_port,
                self._keepalive
            )
            self._client.loop_start()
            logger.info("MQTT connection initiated to %s:%d", self._broker_host, self._broker_port)
            return True
        except Exception as e:
            logger.error("MQTT connection failed: %s", e)
            with self._lock:
                self._state = MQTT_ERROR
            return False

    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Block until connected or timeout. Returns connection state."""
        return self._connected_event.wait(timeout=timeout)

    def publish(
        self,
        topic: str,
        payload: Any,
        qos: Optional[int] = None,
        retain: bool = False
    ) -> bool:
        """
        Publish a message to a topic.
        If disconnected, queues the message for later delivery.
        Payload can be a dict (serialized to JSON) or a string.
        """
        if qos is None:
            qos = self._qos

        if isinstance(payload, (dict, list)):
            try:
                message: str = json.dumps(payload)
            except (TypeError, ValueError) as e:
                logger.error("Failed to serialize payload for %s: %s", topic, e)
                return False
        else:
            message = str(payload)

        with self._lock:
            if self._state != MQTT_CONNECTED:
                if len(self._message_queue) < self._max_queue_size:
                    self._message_queue.append((topic, message, qos))
                    logger.debug("MQTT offline — queued message for %s (%d in queue)",
                                 topic, len(self._message_queue))
                else:
                    logger.warning("MQTT queue full — dropping message for %s", topic)
                return False

        try:
            result = self._client.publish(topic, message, qos=qos, retain=retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug("Published to %s", topic)
                return True
            else:
                logger.warning("MQTT publish failed for %s: rc=%d", topic, result.rc)
                return False
        except Exception as e:
            logger.error("MQTT publish error for %s: %s", topic, e)
            return False

    def publish_health(self, data: dict[str, Any]) -> bool:
        """Convenience: publish health telemetry."""
        topic: str = self._topics.get("health", "cerberus/telemetry/health")
        return self.publish(topic, data)

    def publish_sensor(self, data: dict[str, Any]) -> bool:
        """Convenience: publish sensor reading."""
        topic: str = self._topics.get("sensors", "cerberus/telemetry/sensors")
        return self.publish(topic, data)

    def publish_detection(self, detection_type: str, data: dict[str, Any]) -> bool:
        """Convenience: publish AI detection."""
        base: str = self._topics.get("detections", "cerberus/detections")
        topic: str = f"{base}/{detection_type}"
        return self.publish(topic, data)

    def publish_mission_status(self, data: dict[str, Any]) -> bool:
        """Convenience: publish mission status update."""
        topic: str = self._topics.get("mission_status", "cerberus/mission/status")
        return self.publish(topic, data)

    def publish_alert(self, alert: dict[str, Any]) -> bool:
        """Convenience: publish critical alert."""
        topic: str = self._topics.get("alerts", "cerberus/alerts")
        return self.publish(topic, alert, qos=2)

    def register_command_handler(
        self,
        topic_pattern: str,
        handler: Callable[[MQTTMessage], None]
    ) -> None:
        """
        Register a callback for inbound command messages.
        Supports MQTT wildcards: + (single level), # (multi level).

        Usage:
            mqtt.register_command_handler("cerberus/command/drive", handle_drive)
            mqtt.register_command_handler("cerberus/command/#", handle_all)
        """
        with self._lock:
            if topic_pattern not in self._command_handlers:
                self._command_handlers[topic_pattern] = []
            self._command_handlers[topic_pattern].append(handler)
        logger.info("Registered command handler for %s", topic_pattern)

    def unregister_command_handler(
        self,
        topic_pattern: str,
        handler: Callable[[MQTTMessage], None]
    ) -> None:
        """Remove a specific command handler."""
        with self._lock:
            if topic_pattern in self._command_handlers:
                try:
                    self._command_handlers[topic_pattern].remove(handler)
                except ValueError:
                    pass

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._state == MQTT_CONNECTED

    @property
    def state(self) -> int:
        with self._lock:
            return self._state

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._message_queue)

    def disconnect(self) -> None:
        """Cleanly disconnect from the MQTT broker."""
        try:
            self._client.loop_stop()
            self._client.disconnect()
            with self._lock:
                self._state = MQTT_DISCONNECTED
            self._connected_event.clear()
            logger.info("MQTT client disconnected")
        except Exception as e:
            logger.error("Error during MQTT disconnect: %s", e)

    def __repr__(self) -> str:
        states: dict[int, str] = {
            MQTT_CONNECTED: "connected",
            MQTT_DISCONNECTED: "disconnected",
            MQTT_ERROR: "error"
        }
        return (
            f"CerberusMQTT(broker='{self._broker_host}:{self._broker_port}', "
            f"state={states.get(self._state, 'unknown')}, "
            f"queued={self.queued_count})"
        )