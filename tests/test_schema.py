import json

import jsonschema
import pytest
from pydantic import ValidationError

from schema.export import EXAMPLES_DIR, JSON_DIR, MESSAGE_MODELS, example_messages, json_schemas
from schema.family import load_family
from schema.models import NodeStateMessage, VoiceReply

FAMILY = {"MEMBER_A", "MEMBER_B", "MEMBER_C", "MEMBER_D"}


def _schema_name(model) -> str:
    return next(name for name, cls in MESSAGE_MODELS.items() if isinstance(model, cls))


@pytest.mark.parametrize("name", list(example_messages()))
def test_example_passes_json_schema(name):
    message = example_messages()[name]
    payload = message.model_dump(mode="json")
    jsonschema.validate(payload, json_schemas()[_schema_name(message)])
    # JSON으로 왕복해도 같은 메시지
    assert type(message).model_validate_json(json.dumps(payload), context={"family": FAMILY}) == message


def test_generated_files_are_up_to_date():
    """모델을 바꾸고 python -m schema.export를 안 돌렸으면 실패."""
    for name, schema in json_schemas().items():
        assert json.loads((JSON_DIR / f"{name}.schema.json").read_text(encoding="utf-8")) == schema
    for name, message in example_messages().items():
        assert json.loads((EXAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8")) == message.model_dump(mode="json")


def _resting() -> dict:
    return example_messages()["node_state_resting"].model_dump(mode="json")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["persons"][0].update(identity_confidence=1.5),
        lambda m: m["persons"][0].update(posture="SLEEPING"),
        lambda m: m["persons"][0].update(identity="MEMBER_X"),
        lambda m: m["persons"][0].update(identity="UNKNOWN"),  # source는 FACE인데 identity만 UNKNOWN
        lambda m: m.update(event_type="HEARTBEAT"),            # HEARTBEAT인데 changes가 있음
        lambda m: m.update(changes=[]),                        # STATE_CHANGE인데 changes가 없음
        lambda m: m.update(changes=[{"type": "LIGHT_CHANGED", "track_id": 1}]),
        lambda m: m.update(changes=[{"type": "POSTURE_CHANGED"}]),
        lambda m: m.update(occupancy={"state": "EMPTY", "confidence": 0.9}),
        lambda m: m.update(persons=m["persons"] * 2),          # track_id 중복
        lambda m: m.update(timestamp="2026-10-30T22:00:00"),   # 시간대 없음
        lambda m: m.update(extra_field=1),
        lambda m: m.update(schema_version="0.9"),  # 모르는 버전
    ],
)
def test_invalid_node_state_rejected(mutate):
    message = _resting()
    mutate(message)
    with pytest.raises(ValidationError):
        NodeStateMessage.model_validate(message, context={"family": FAMILY})


def test_voice_reply_no_response_must_have_null_text():
    reply = example_messages()["voice_reply"].model_dump(mode="json")
    reply["no_response"] = True
    with pytest.raises(ValidationError):
        VoiceReply.model_validate(reply)


def test_family_file_loads():
    assert load_family() == FAMILY


@pytest.mark.parametrize("members", [["UNKNOWN"], ["GUEST"], ["member_a"], ["MEMBER_A", "MEMBER_A"]])
def test_family_rejects_bad_names(tmp_path, members):
    path = tmp_path / "family.yaml"
    path.write_text(json.dumps({"members": members}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_family(path)


def test_boot_id_is_fixed_per_program_and_per_fake_node():
    """boot_id: 한 번 켜진 동안은 같고, 가짜 송신기는 방마다 따로 (2026-11-19, 이효창 요청)."""
    from uuid import uuid4

    from fusion.events import EventBuilder, RoomSnapshot
    from schema.models import BOOT_ID, SCHEMA_VERSION, NodeStatus, VoiceCommand

    snap = RoomSnapshot.model_validate({"occupancy": {"state": "EMPTY", "confidence": 0.9}, "persons": [],
                                        "audio": {"context": "SILENT", "confidence": 0.8}, "light": "BRIGHT"})
    clock = iter(range(100)).__next__
    builder = EventBuilder("node_room1", "room1", FAMILY, clock=lambda: clock())
    first, second = builder.update(snap), builder.update(snap)
    assert first.boot_id == second.boot_id == BOOT_ID and first.schema_version == SCHEMA_VERSION == "0.2"
    other = uuid4()
    assert EventBuilder("node_room2", "room2", FAMILY, boot_id=other).update(snap).boot_id == other
    assert VoiceCommand(node_id="node_room1", room_id="room1", text="불 켜줘", confidence=0.8).boot_id == BOOT_ID
    status = NodeStatus(node_id="node_room1", room_id="room1", online=False)
    assert (status.type, status.online, status.boot_id) == ("NODE_STATUS", False, BOOT_ID)


def test_speak_command_accepts_old_version_from_central():
    """중앙(송정빈)이 아직 0.1로 보내도 Node는 받는다."""
    from schema.models import SpeakCommand

    assert SpeakCommand.model_validate({"schema_version": "0.1", "text": "주무실 건가요?", "expect_reply": True})
