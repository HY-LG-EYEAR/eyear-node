"""메시지 출력 대상: MQTT, 화면(stdout), JSONL 파일. 여러 개를 동시에 쓸 수 있다."""

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import paho.mqtt.client as mqtt
from pydantic import BaseModel

from schema.enums import EventType
from schema.models import NodeStateMessage, SpeakCommand, VoiceCommand, VoiceReply


class Sink(Protocol):
    def publish(self, topic: str, message: BaseModel) -> None: ...
    def close(self) -> None: ...


def summarize(message: BaseModel) -> str:
    """사람이 읽기 위한 한 줄 요약."""
    if isinstance(message, NodeStateMessage):
        changes = ", ".join(
            c.type if c.track_id is None else f"{c.type}#{c.track_id}" for c in message.changes
        )
        persons = "; ".join(
            f"#{p.track_id} {p.identity}/{p.identity_source} {p.posture} {p.motion} {p.activity}"
            for p in message.persons
        ) or "-"
        head = f"[{message.room_id} #{message.seq} {message.event_type}]"
        if changes:
            head += f" {changes}"
        return (
            f"{head} | occ={message.occupancy.state} | {persons}"
            f" | audio={message.audio.context} light={message.light}"
        )
    if isinstance(message, VoiceReply):
        said = "(응답 없음)" if message.no_response else f"'{message.text}' ({message.confidence:.2f})"
        return f"[{message.room_id} VOICE_REPLY] {said}"
    if isinstance(message, VoiceCommand):
        return f"[{message.room_id} VOICE_COMMAND] '{message.text}' ({message.confidence:.2f})"
    if isinstance(message, SpeakCommand):
        return f"[SPEAK] '{message.text}' expect_reply={message.expect_reply}"
    return message.model_dump_json()


class StdoutSink:
    """summary: 변화·음성만 한 줄로 / all: heartbeat 포함 한 줄로 / full: 전체 JSON."""

    def __init__(self, mode: Literal["summary", "all", "full"] = "summary"):
        self.mode = mode

    def publish(self, topic: str, message: BaseModel) -> None:
        if self.mode == "full":
            print(topic, message.model_dump_json(indent=2), sep="\n", flush=True)
            return
        if (
            self.mode == "summary"
            and isinstance(message, NodeStateMessage)
            and message.event_type == EventType.HEARTBEAT
        ):
            return
        print(summarize(message), flush=True)

    def close(self) -> None:
        pass


class JsonlSink:
    """한 줄에 {"topic", "payload"} 하나. 재생(recorder)과 팀원 오프라인 테스트용."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def publish(self, topic: str, message: BaseModel) -> None:
        record = {"topic": topic, "payload": message.model_dump(mode="json")}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


MessageHandler = Callable[[str, bytes], None]


class MqttSink:
    def __init__(self, host: str = "localhost", port: int = 1883):
        self._handlers: dict[str, MessageHandler] = {}
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(host, port)
        except OSError as e:
            raise ConnectionError(
                f"MQTT 브로커 {host}:{port}에 연결할 수 없음. mosquitto가 실행 중인지 확인 "
                "(brew services start mosquitto)"
            ) from e
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        # 재연결 시에도 구독 유지
        for topic in self._handlers:
            client.subscribe(topic, qos=1)

    def _on_message(self, client, userdata, msg):
        for pattern, handler in self._handlers.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                try:
                    handler(msg.topic, msg.payload)
                except Exception as e:  # 잘못된 메시지 하나로 송신기가 멈추지 않게
                    print(f"[오류] {msg.topic} 처리 실패: {e}", file=sys.stderr, flush=True)

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        self._handlers[topic] = handler
        self._client.subscribe(topic, qos=1)

    def publish(self, topic: str, message: BaseModel) -> None:
        self._client.publish(topic, message.model_dump_json(), qos=1)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


class MultiSink:
    """여러 스레드(카메라 루프, 음성 대화)에서 동시에 불러도 메시지가 섞이지 않게 잠근다."""

    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks
        self._lock = threading.Lock()

    def publish(self, topic: str, message: BaseModel) -> None:
        with self._lock:
            for sink in self.sinks:
                sink.publish(topic, message)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
