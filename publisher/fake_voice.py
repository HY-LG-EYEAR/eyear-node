"""가짜 음성 Node: 중앙의 SpeakCommand를 받아 화면에 출력하고, 정해 둔 대답을 VoiceReply로 보낸다.

마이크·스피커 없이 중앙 에이전트(송정빈)의 질문 → 대답 흐름을 테스트하기 위한 것.
가짜 송신기(fake_sender)에 포함되어 함께 돌며, 따로 실행할 수도 있다.

    python -m publisher.fake_voice --mqtt localhost --room room1 --reply 응
    python -m publisher.fake_voice --mqtt localhost --room room1 --reply none   # 응답 없음
"""

import argparse
import threading
import time

from publisher import topics
from publisher.sinks import MqttSink, Sink, summarize
from schema.models import SpeakCommand, VoiceReply

ANSWER_DELAY = 1.5  # 출력이 끝난 뒤 대답하기까지 걸리는 시간(초), 흉내만 냄


class VoiceResponder:
    def __init__(
        self,
        mqtt: MqttSink,
        out: Sink,
        node_ids: dict[str, str],
        reply: str | None = "응",
        confidence: float = 0.8,
    ):
        """node_ids: {room_id: node_id}. 여기 있는 방의 SpeakCommand에만 대답한다. reply=None이면 응답 없음."""
        self.out = out
        self.node_ids = node_ids
        self.reply = reply
        self.confidence = confidence
        mqtt.subscribe(topics.SPEAK_ALL_ROOMS, self._on_speak)

    def _on_speak(self, topic: str, payload: bytes) -> None:
        room_id = topics.room_of(topic)
        if room_id not in self.node_ids:
            return
        command = SpeakCommand.model_validate_json(payload)
        print(f"🔊 [{room_id}] {summarize(command)}", flush=True)
        if command.expect_reply:
            wait = command.listen_seconds if self.reply is None else ANSWER_DELAY
            threading.Timer(wait, self._send_reply, args=(room_id, command)).start()

    def _send_reply(self, room_id: str, command: SpeakCommand) -> None:
        answered = self.reply is not None
        reply = VoiceReply(
            node_id=self.node_ids[room_id],
            room_id=room_id,
            command_id=command.command_id,
            text=self.reply,
            confidence=self.confidence if answered else None,
            no_response=not answered,
        )
        self.out.publish(topics.voice_reply(room_id), reply)


def parse_reply(value: str) -> str | None:
    return None if value.lower() == "none" else value


def main() -> None:
    parser = argparse.ArgumentParser(description="가짜 음성 Node")
    parser.add_argument("--mqtt", default="localhost", help="브로커 HOST[:PORT]")
    parser.add_argument("--room", action="append", default=None, help="대답할 방 (여러 번 지정 가능, 기본 room1)")
    parser.add_argument("--reply", default="응", help='대답 문장. "none"이면 응답 없음')
    args = parser.parse_args()

    host, _, port = args.mqtt.partition(":")
    mqtt = MqttSink(host, int(port or 1883))
    rooms = args.room or ["room1"]
    VoiceResponder(mqtt, mqtt, {room: f"node_{room}" for room in rooms}, parse_reply(args.reply))
    print(f"가짜 음성 Node 대기 중: {', '.join(rooms)} (Ctrl+C로 종료)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        mqtt.close()


if __name__ == "__main__":
    main()
