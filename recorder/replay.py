"""녹화 재생 (R-23): Node가 보낸 메시지 기록(JSONL)을 같은 시간 간격으로 다시 보낸다.

    # 녹화: 실제 Node를 켤 때 --jsonl로 보낸 메시지를 저장 (원본 영상·음성이 아니라 의미 정보만)
    python main.py --mqtt localhost --jsonl recordings/오늘.jsonl
    # 재생
    python -m recorder.replay recorder/samples/2026-11-19_room1_rest_passerby.jsonl --mqtt localhost
    python -m recorder.replay 기록.jsonl --mqtt localhost --speed 2 --from 60 --to 150 --loop

- 방 상태(.../state)만 다시 보낸다. 음성 대답은 원래 질문(command_id)에 묶여 있어서 기본으로 빼고,
  "하이 엘지" 뒤 명령(.../voice/command)은 --voice일 때만 보낸다. 켜짐·꺼짐(.../status)은 재생기가 새로 보낸다.
- 받는 쪽이 지금 막 켜진 Node로 보도록 고쳐서 보낸다: 재생할 때마다 새 boot_id, seq는 0부터,
  event_id는 새로, timestamp는 지금 시각 기준 (기록 사이 간격은 그대로).
- 원래 내용(재실·사람·소리·밝기·changes)은 그대로.
"""

import argparse
import json
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from publisher import topics
from publisher.sinks import MqttSink, MultiSink, NodeStatusPublisher, Sink, StdoutSink, parse_broker
from schema.models import NodeStateMessage, VoiceCommand


@dataclass(frozen=True)
class Record:
    at: float          # 기록 시작부터 몇 초 뒤
    topic: str
    payload: dict


def load(path: Path | str, voice: bool = False) -> list[Record]:
    """기록 파일 → 재생할 메시지 (시간순). 상태(와 --voice면 음성 명령)만."""
    keep = ("/state", "/voice/command") if voice else ("/state",)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["topic"].endswith(keep):
            rows.append((datetime.fromisoformat(r["payload"]["timestamp"]), r["topic"], r["payload"]))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return []
    t0 = rows[0][0]
    return [Record((t - t0).total_seconds(), topic, payload) for t, topic, payload in rows]


def window(records: list[Record], start: float | None, end: float | None) -> list[Record]:
    """start~end초 구간만, 시간은 구간 시작이 0이 되게."""
    picked = [r for r in records if (start is None or r.at >= start) and (end is None or r.at <= end)]
    if not picked:
        return []
    base = picked[0].at
    return [Record(r.at - base, r.topic, r.payload) for r in picked]


class Rewriter:
    """기록된 메시지를 '지금 켜진 Node'의 메시지로 고친다 (Node별 새 boot_id, seq 0부터, 새 event_id, 지금 시각)."""

    def __init__(self, started: datetime):
        self.started = started
        self.boot_ids: dict[str, UUID] = {}
        self._seq: dict[str, int] = {}

    def boot_id(self, node_id: str) -> UUID:
        return self.boot_ids.setdefault(node_id, uuid4())

    def rewrite(self, record: Record, offset: float):
        p = dict(record.payload)
        node = p["node_id"]
        p.update(event_id=str(uuid4()), boot_id=str(self.boot_id(node)),
                 timestamp=(self.started + timedelta(seconds=offset)).isoformat())
        if record.topic.endswith("/state"):
            p["seq"] = self._seq.get(node, 0)
            self._seq[node] = p["seq"] + 1
            return NodeStateMessage.model_validate(p)
        return VoiceCommand.model_validate(p)


def play(records: list[Record], sink: Sink, speed: float = 1.0, loop: bool = False,
         on_start=None, clock=time.monotonic, sleep=time.sleep, now=datetime.now) -> None:
    """records를 시간 간격대로 sink로 보낸다. loop면 끝나고 다시 (새 boot_id로)."""
    while True:
        started_wall = now().astimezone()
        rw = Rewriter(started_wall)
        if on_start:
            on_start(rw)
        t0 = clock()
        for r in records:
            wait = r.at / speed - (clock() - t0)
            if wait > 0:
                sleep(wait)
            sink.publish(r.topic, rw.rewrite(r, r.at / speed))
        if not loop:
            return


def _stop(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 녹화 재생 (Node 메시지 기록을 다시 보냄)")
    parser.add_argument("recording", help="Node가 --jsonl로 남긴 기록 파일")
    parser.add_argument("--mqtt", default=None, metavar="HOST[:PORT]", help="없으면 화면 출력만")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 (2 = 두 배 빠르게)")
    parser.add_argument("--from", dest="start", type=float, default=None, metavar="초", help="기록의 이 시점부터")
    parser.add_argument("--to", dest="end", type=float, default=None, metavar="초", help="기록의 이 시점까지")
    parser.add_argument("--loop", action="store_true", help="끝나면 처음부터 반복 (매번 새 boot_id)")
    parser.add_argument("--voice", action="store_true", help='"하이 엘지" 뒤 음성 명령도 보냄')
    parser.add_argument("--stdout", choices=["summary", "all", "full", "off"], default="summary")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)

    records = window(load(args.recording, args.voice), args.start, args.end)
    if not records:
        parser.error("재생할 메시지가 없음 (파일·구간 확인)")
    rooms = sorted({r.payload["room_id"]: r.payload["node_id"] for r in records}.items())
    print(f"재생: {len(records)}개, {records[-1].at:.0f}초 분량, 방 {', '.join(r for r, _ in rooms)} "
          f"(속도 ×{args.speed})", flush=True)

    sinks: list[Sink] = [StdoutSink(args.stdout)] if args.stdout != "off" else []
    mqtt = MqttSink(*parse_broker(args.mqtt)) if args.mqtt else None
    if mqtt:
        sinks.append(mqtt)
    sink = MultiSink(sinks)
    statuses: list[NodeStatusPublisher] = []

    def on_start(rw: Rewriter) -> None:
        for s in statuses:
            s.close()
        statuses.clear()
        if args.mqtt:
            host, port = parse_broker(args.mqtt)
            statuses.extend(NodeStatusPublisher(host, port, node, room, rw.boot_id(node)) for room, node in rooms)

    try:
        play(records, sink, args.speed, args.loop, on_start)
        print("재생 끝", flush=True)
    except KeyboardInterrupt:
        print("\n종료", flush=True)
    finally:
        for s in statuses:
            s.close()
        sink.close()


if __name__ == "__main__":
    main()
