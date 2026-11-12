"""pydantic 모델에서 JSON Schema 파일과 예시 메시지를 생성한다.

    python -m schema.export

결과
- schema/json/<메시지>.schema.json : 팀원에게 전달하는 형식 정의
- schema/examples/<예시>.json      : 실제 메시지 예시

주의: 필드 사이의 규칙(예: HEARTBEAT에는 changes가 없음)은 JSON Schema로 표현되지 않는다.
docs/schema_v0.md에 따로 적는다.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from schema.models import (
    NodeStateMessage,
    SpeakCommand,
    VoiceCommand,
    VoiceReply,
)

SCHEMA_DIR = Path(__file__).resolve().parent
JSON_DIR = SCHEMA_DIR / "json"
EXAMPLES_DIR = SCHEMA_DIR / "examples"

MESSAGE_MODELS: dict[str, type[BaseModel]] = {
    "node_state": NodeStateMessage,
    "speak_command": SpeakCommand,
    "voice_reply": VoiceReply,
    "voice_command": VoiceCommand,
}

_KST = timezone(timedelta(hours=9))
_T0 = datetime(2026, 10, 30, 22, 0, 0, tzinfo=_KST)
_QUESTION_ID = UUID("00000000-0000-4000-8000-0000000000c1")


def _uuid(n: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{n:012d}")


def _person(**overrides) -> dict:
    person = {
        "track_id": 1,
        "identity": "MEMBER_A",
        "identity_confidence": 0.91,
        "identity_source": "FACE",
        "posture": "STANDING",
        "posture_confidence": 0.88,
        "motion": "ACTIVE",
        "activity": "OTHER",
        "activity_confidence": 0.85,
        "position": {"x": 1.40, "y": 1.25},
    }
    return person | overrides


def _state(n: int, seconds: float, **fields) -> dict:
    return {
        "schema_version": "0.1",
        "event_id": _uuid(n),
        "seq": n,
        "node_id": "node_room1",
        "room_id": "room1",
        "timestamp": _T0 + timedelta(seconds=seconds),
    } | fields


def example_messages() -> dict[str, BaseModel]:
    """시나리오(방1, A의 휴식·수면) 흐름을 따라가는 예시 메시지."""
    bright_room = {"audio": {"context": "SILENT", "confidence": 0.80}, "light": "BRIGHT"}
    raw = {
        # t1: 밝은 방1에 A가 들어옴 → 얼굴로 식별
        "node_state_person_entered": (NodeStateMessage, _state(
            1, 0,
            event_type="STATE_CHANGE",
            changes=[
                {"type": "PERSON_ENTERED", "track_id": 1},
                {"type": "OCCUPANCY_CHANGED"},
            ],
            occupancy={"state": "OCCUPIED", "confidence": 0.95},
            persons=[_person()],
            **bright_room,
        )),
        # t2: 침대에 누움 → RESTING (중앙이 수면 질문을 만드는 계기)
        "node_state_resting": (NodeStateMessage, _state(
            2, 40,
            event_type="STATE_CHANGE",
            changes=[
                {"type": "POSTURE_CHANGED", "track_id": 1},
                {"type": "ACTIVITY_CHANGED", "track_id": 1},
            ],
            occupancy={"state": "OCCUPIED", "confidence": 0.97},
            persons=[_person(
                posture="LYING", posture_confidence=0.84, motion="SLIGHT",
                activity="RESTING", activity_confidence=0.80, position={"x": 2.40, "y": 2.67},
            )],
            **bright_room,
        )),
        # t4: 소등 후 어두움 → 레이더로 신원 이어받기, 신뢰도 낮춤. 자세는 볼 수 없어 UNKNOWN, RESTING은 유지
        "node_state_dark_track_carried": (NodeStateMessage, _state(
            3, 70,
            event_type="STATE_CHANGE",
            changes=[
                {"type": "LIGHT_CHANGED"},
                {"type": "IDENTITY_CHANGED", "track_id": 1},
                {"type": "POSTURE_CHANGED", "track_id": 1},
            ],
            occupancy={"state": "OCCUPIED", "confidence": 0.85},
            persons=[_person(
                identity_confidence=0.75, identity_source="TRACK_CARRIED",
                posture="UNKNOWN", posture_confidence=0.0, motion="STILL",
                activity="RESTING", activity_confidence=0.70, position={"x": 2.40, "y": 2.67},
            )],
            audio={"context": "SILENT", "confidence": 0.85},
            light="DARK",
        )),
        # 위치 추적(LD2450)은 놓쳤지만 재실 센서(LD2410C)는 사람을 감지
        "node_state_position_lost": (NodeStateMessage, _state(
            4, 72,
            event_type="HEARTBEAT",
            occupancy={"state": "OCCUPIED", "confidence": 0.90},
            persons=[],
            audio={"context": "SILENT", "confidence": 0.85},
            light="DARK",
        )),
        # 센서가 없거나 고장: 지어내지 않고 UNKNOWN (R-09, R-13)
        "node_state_sensors_unavailable": (NodeStateMessage, _state(
            5, 74,
            event_type="HEARTBEAT",
            occupancy={"state": "UNKNOWN", "confidence": 0.0},
            persons=[],
            audio={"context": "UNKNOWN", "confidence": 0.0},
            light="UNKNOWN",
        )),
        # 처음부터 어두운 방에 들어온 사람: 신원 UNKNOWN, 레이더 위치만 있음
        "node_state_unknown_person": (NodeStateMessage, _state(
            6, 76,
            event_type="STATE_CHANGE",
            changes=[{"type": "PERSON_ENTERED", "track_id": 2}],
            occupancy={"state": "OCCUPIED", "confidence": 0.92},
            persons=[_person(
                track_id=2, identity="UNKNOWN", identity_confidence=0.0, identity_source="UNKNOWN",
                posture="UNKNOWN", posture_confidence=0.0, motion="ACTIVE",
                activity="UNKNOWN", activity_confidence=0.0, position={"x": 0.40, "y": 1.25},
            )],
            audio={"context": "SILENT", "confidence": 0.85},
            light="DARK",
        )),
        # t2 → 중앙이 보내는 질문
        "speak_command_question": (SpeakCommand, {
            "schema_version": "0.1",
            "command_id": _QUESTION_ID,
            "timestamp": _T0 + timedelta(seconds=42),
            "text": "A님, 주무실 건가요?",
            "expect_reply": True,
            "listen_seconds": 5,
        }),
        # t5 → 중앙이 보내는 대답 (대답을 기다리지 않음)
        "speak_command_announcement": (SpeakCommand, {
            "schema_version": "0.1",
            "command_id": _uuid(201),
            "timestamp": _T0 + timedelta(seconds=301),
            "text": "네, 불을 켤게요",
            "expect_reply": False,
            "listen_seconds": 5,
        }),
        # t3 → Node가 들은 대답
        "voice_reply": (VoiceReply, {
            "schema_version": "0.1",
            "event_id": _uuid(101),
            "node_id": "node_room1",
            "room_id": "room1",
            "timestamp": _T0 + timedelta(seconds=46),
            "type": "VOICE_REPLY",
            "command_id": _QUESTION_ID,
            "text": "응",
            "confidence": 0.72,
            "no_response": False,
        }),
        "voice_reply_no_response": (VoiceReply, {
            "schema_version": "0.1",
            "event_id": _uuid(102),
            "node_id": "node_room1",
            "room_id": "room1",
            "timestamp": _T0 + timedelta(seconds=49),
            "type": "VOICE_REPLY",
            "command_id": _QUESTION_ID,
            "text": None,
            "confidence": None,
            "no_response": True,
        }),
        # t5 → "하이 엘지, 불 켜줘"
        "voice_command": (VoiceCommand, {
            "schema_version": "0.1",
            "event_id": _uuid(103),
            "node_id": "node_room1",
            "room_id": "room1",
            "timestamp": _T0 + timedelta(seconds=300),
            "type": "VOICE_COMMAND",
            "text": "불 켜줘",
            "confidence": 0.81,
        }),
    }
    return {name: model.model_validate(data) for name, (model, data) in raw.items()}


def json_schemas() -> dict[str, dict]:
    return {
        name: model.model_json_schema(mode="serialization")
        for name, model in MESSAGE_MODELS.items()
    }


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    JSON_DIR.mkdir(exist_ok=True)
    EXAMPLES_DIR.mkdir(exist_ok=True)
    for name, schema in json_schemas().items():
        _write(JSON_DIR / f"{name}.schema.json", schema)
    for name, message in example_messages().items():
        _write(EXAMPLES_DIR / f"{name}.json", message.model_dump(mode="json"))
    print(f"JSON Schema → {JSON_DIR}")
    print(f"예시 메시지 → {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()
