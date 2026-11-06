"""가짜 송신기: 실제 센서 없이 스키마 v0 형식의 Node 메시지를 보낸다.

    python -m publisher.fake_sender                       # 시나리오 재생, 5초마다 다음 단계, 화면 출력만
    python -m publisher.fake_sender --mqtt localhost      # MQTT로 전송 + 중앙의 질문에 가짜 대답
    python -m publisher.fake_sender --step                # Enter를 누를 때마다 다음 단계
    python -m publisher.fake_sender --mode random --seed 1
    python -m publisher.fake_sender --jsonl out/run.jsonl # 파일로도 기록

모든 방이 heartbeat_interval마다 HEARTBEAT를 내고, 상태가 바뀌면 즉시 STATE_CHANGE를 낸다.
"""

import argparse
import queue
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fusion.events import EventBuilder, RoomSnapshot
from publisher import topics
from publisher.fake_voice import VoiceResponder, parse_reply
from publisher.sinks import JsonlSink, MqttSink, MultiSink, Sink, StdoutSink
from schema.enums import Activity, AudioContext, IdentitySource, Light, Motion, OccupancyState, Posture
from schema.family import load_family
from schema.models import VoiceCommand

DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios" / "summer_evening.yaml"
TICK = 0.1


# ---------- 시나리오 ----------

@dataclass
class Step:
    id: str
    title: str
    expect: str
    rooms: dict[str, RoomSnapshot]
    voice_commands: list[dict] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    nodes: dict[str, str]  # room_id → node_id
    steps: list[Step]


def load_scenario(path: Path | str, family: frozenset[str]) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    nodes: dict[str, str] = data["nodes"]
    steps = []
    for raw in data["steps"]:
        rooms = raw.get("rooms") or {}
        unknown_rooms = rooms.keys() - nodes.keys()
        if unknown_rooms:
            raise ValueError(f"{raw['id']}: nodes에 없는 방 {sorted(unknown_rooms)}")
        for command in raw.get("voice_commands") or []:
            if command["room"] not in nodes:
                raise ValueError(f"{raw['id']}: nodes에 없는 방 {command['room']}")
        steps.append(Step(
            id=raw["id"],
            title=raw.get("title", ""),
            expect=raw.get("expect", ""),
            rooms={
                room: RoomSnapshot.model_validate(snapshot, context={"family": family})
                for room, snapshot in rooms.items()
            },
            voice_commands=raw.get("voice_commands") or [],
        ))
    if not steps or steps[0].rooms.keys() != nodes.keys():
        raise ValueError("첫 단계에는 모든 방의 상태가 있어야 함")
    return Scenario(name=data.get("name", Path(path).stem), nodes=nodes, steps=steps)


class ScenarioPlayer:
    def __init__(self, scenario: Scenario, sink: Sink, interval: float, manual: bool, loop: bool, voice_enabled: bool):
        self.scenario = scenario
        self.sink = sink
        self.interval = interval
        self.manual = manual
        self.loop = loop
        self.voice_enabled = voice_enabled
        self.snapshots: dict[str, RoomSnapshot] = {}
        self._next = 0
        self._last_advance = 0.0
        self._end_pending = False
        self._enter: queue.Queue[str] = queue.Queue()
        if manual:
            threading.Thread(target=self._read_enter, daemon=True).start()

    def _read_enter(self) -> None:
        for line in sys.stdin:
            self._enter.put(line)

    def tick(self, now: float) -> None:
        if self._end_pending:
            # 마지막 단계의 메시지가 나간 다음 틱에 출력
            print("\n━━ 시나리오 끝. 마지막 상태로 heartbeat 계속 (Ctrl+C로 종료)", flush=True)
            self._end_pending = False
        if self._next == 0:
            self._advance(now)  # 첫 단계는 바로 시작
        elif self.manual:
            if not self._enter.empty():
                self._enter.get()
                self._advance(now)
        elif now - self._last_advance >= self.interval:
            self._advance(now)

    def _advance(self, now: float) -> None:
        steps = self.scenario.steps
        if self._next >= len(steps):
            if not self.loop:
                return
            self._next = 0
        step = steps[self._next]
        self._next += 1
        self._last_advance = now

        print(f"\n━━ {step.id}  {step.title}", flush=True)
        if step.expect:
            print(f"   기대: {step.expect}", flush=True)
        for command in step.voice_commands:
            room = command["room"]
            self.sink.publish(topics.voice_command(room), VoiceCommand(
                node_id=self.scenario.nodes[room],
                room_id=room,
                text=command["text"],
                confidence=command["confidence"],
            ))
        if step.id == "t3" and not self.voice_enabled:
            print("   (MQTT 없이 실행 중이라 질문을 받을 수 없음 → 대답 없음)", flush=True)
        self.snapshots.update(step.rooms)

        if self._next >= len(steps) and not self.loop:
            self._end_pending = True
        elif self.manual:
            print("   [Enter] 다음 단계", flush=True)


# ---------- 랜덤 ----------

class RandomWalker:
    """스키마 범위 안에서 임의로 상태를 바꾼다. 받는 쪽 코드의 견고성 테스트용.

    같은 가족이 두 방에 동시에 나타나지 않게 한다. 한 방에 최대 3명 (LD2450 한계).
    """

    def __init__(self, rooms: list[str], family: frozenset[str], interval: float, seed: int | None):
        self.rng = random.Random(seed)
        self.family = sorted(family)
        self.interval = interval
        self._last = 0.0
        self._next_track = {room: 1 for room in rooms}
        self.snapshots = {room: self._empty() for room in rooms}

    def _empty(self) -> RoomSnapshot:
        return RoomSnapshot.model_validate({
            "occupancy": {"state": "EMPTY", "confidence": 0.95},
            "persons": [],
            "audio": {"context": "SILENT", "confidence": 0.8},
            "light": "BRIGHT",
        })

    def _conf(self) -> float:
        return round(self.rng.uniform(0.5, 0.99), 2)

    def _position(self) -> dict | None:
        if self.rng.random() < 0.1:
            return None
        return {"x": round(self.rng.uniform(0, 4), 2), "y": round(self.rng.uniform(0, 4), 2)}

    def _set_posture(self, person: dict, posture: str) -> None:
        person["posture"] = posture
        person["posture_confidence"] = 0.0 if posture == Posture.UNKNOWN else self._conf()
        if posture == Posture.UNKNOWN:
            person["activity"], person["activity_confidence"] = Activity.UNKNOWN, 0.0
        elif posture == Posture.LYING and self.rng.random() < 0.6:
            person["activity"], person["activity_confidence"] = Activity.RESTING, self._conf()
        else:
            person["activity"], person["activity_confidence"] = Activity.OTHER, self._conf()

    def _new_person(self, room: str) -> dict:
        in_use = {p.identity for s in self.snapshots.values() for p in s.persons}
        free = [name for name in self.family if name not in in_use]
        identity = self.rng.choice(free) if free and self.rng.random() < 0.85 else "UNKNOWN"
        person = {
            "track_id": self._next_track[room],
            "identity": identity,
            "identity_confidence": 0.0 if identity == "UNKNOWN" else self._conf(),
            "identity_source": IdentitySource.UNKNOWN if identity == "UNKNOWN" else IdentitySource.FACE,
            "motion": "ACTIVE",
            "position": self._position(),
        }
        self._next_track[room] += 1
        self._set_posture(person, self.rng.choice([Posture.WALKING, Posture.STANDING]))
        return person

    def tick(self, now: float) -> None:
        if now - self._last < self.interval:
            return
        self._last = now
        room = self.rng.choice(list(self.snapshots))
        state = self.snapshots[room].model_dump()
        persons = state["persons"]
        options = ["enter", "light", "audio", "occupancy"]
        if persons:
            options += ["leave", "posture", "posture", "motion", "identity_source", "position"]
        action = self.rng.choice(options)

        if action == "enter" and len(persons) < 3:
            persons.append(self._new_person(room))
        elif action == "leave":
            persons.remove(self.rng.choice(persons))
            if not persons:
                state["occupancy"] = {"state": "EMPTY", "confidence": self._conf()}
        elif action == "posture":
            self._set_posture(self.rng.choice(persons), self.rng.choice(list(Posture)))
        elif action == "motion":
            self.rng.choice(persons)["motion"] = self.rng.choice(list(Motion))
        elif action == "identity_source":
            person = self.rng.choice(persons)
            if person["identity"] != "UNKNOWN":
                person["identity_source"] = self.rng.choice(
                    [IdentitySource.FACE, IdentitySource.TRACK_CARRIED, IdentitySource.VOICE])
                person["identity_confidence"] = self._conf()
        elif action == "position":
            self.rng.choice(persons)["position"] = self._position()
        elif action == "light":
            state["light"] = self.rng.choice(list(Light))
        elif action == "audio":
            context = self.rng.choice(list(AudioContext))
            state["audio"] = {"context": context, "confidence": 0.0 if context == AudioContext.UNKNOWN else self._conf()}
        elif action == "occupancy" and not persons:
            # 사람이 없을 때: 빈 방 / 위치는 놓쳤지만 재실 / 센서 불가
            occ = self.rng.choice([OccupancyState.EMPTY, OccupancyState.OCCUPIED, OccupancyState.UNKNOWN])
            state["occupancy"] = {"state": occ, "confidence": 0.0 if occ == OccupancyState.UNKNOWN else self._conf()}

        if persons:
            state["occupancy"] = {"state": "OCCUPIED", "confidence": self._conf()}
        self.snapshots[room] = RoomSnapshot.model_validate(state)


# ---------- 실행 ----------

def build_sinks(args) -> tuple[MultiSink, MqttSink | None]:
    sinks: list[Sink] = []
    if args.stdout != "off":
        sinks.append(StdoutSink(args.stdout))
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    mqtt = None
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        mqtt = MqttSink(host, int(port or 1883))
        sinks.append(mqtt)
    return MultiSink(sinks), mqtt


def _stop(signum, frame):
    """kill(SIGTERM)로 끝내도 Ctrl+C와 같이 정리하고 종료."""
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 가짜 송신기 (스키마 v0)")
    parser.add_argument("--mode", choices=["scenario", "random"], default="scenario")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="시나리오 YAML 경로")
    parser.add_argument("--interval", type=float, default=None,
                        help="단계(랜덤 모드는 변화) 간격(초). 기본: 시나리오 5, 랜덤 2")
    parser.add_argument("--step", action="store_true", help="시나리오를 Enter로 한 단계씩 진행")
    parser.add_argument("--loop", action="store_true", help="시나리오가 끝나면 처음부터 반복")
    parser.add_argument("--heartbeat", type=float, default=1.0, help="heartbeat 주기(초)")
    parser.add_argument("--rooms", default="room1,room2,living,laundry", help="랜덤 모드의 방 목록")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 모드 시드")
    parser.add_argument("--mqtt", default=None, metavar="HOST[:PORT]", help="MQTT 브로커 (예: localhost)")
    parser.add_argument("--jsonl", default=None, metavar="PATH", help="JSONL 파일로도 기록")
    parser.add_argument("--stdout", choices=["summary", "all", "full", "off"], default="summary",
                        help="화면 출력: summary=변화만 / all=heartbeat 포함 / full=전체 JSON")
    parser.add_argument("--reply", default="응", help='가짜 음성 대답 (MQTT 사용 시). "none"이면 응답 없음')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)

    family = load_family()
    sink, mqtt = build_sinks(args)

    if args.mode == "scenario":
        scenario = load_scenario(args.scenario, family)
        nodes = scenario.nodes
        driver = ScenarioPlayer(scenario, sink, args.interval or 5.0, args.step, args.loop, mqtt is not None)
    else:
        rooms = [r.strip() for r in args.rooms.split(",") if r.strip()]
        nodes = {room: f"node_{room}" for room in rooms}
        driver = RandomWalker(rooms, family, args.interval or 2.0, args.seed)

    if mqtt:
        VoiceResponder(mqtt, sink, nodes, parse_reply(args.reply))
        print(f"MQTT 전송 중: {topics.PREFIX}/<room>/state (질문 수신: {topics.SPEAK_ALL_ROOMS})", flush=True)

    builders = {
        room: EventBuilder(node_id, room, family, args.heartbeat)
        for room, node_id in nodes.items()
    }
    try:
        while True:
            driver.tick(time.monotonic())
            for room, builder in builders.items():
                if room in driver.snapshots:
                    message = builder.update(driver.snapshots[room])
                    if message:
                        sink.publish(topics.state(room), message)
            time.sleep(TICK)
    except KeyboardInterrupt:
        print("\n종료", flush=True)
    finally:
        sink.close()


if __name__ == "__main__":
    main()
