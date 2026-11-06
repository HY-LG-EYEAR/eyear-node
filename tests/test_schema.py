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
        lambda m: m.update(schema_version="0.2"),
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
