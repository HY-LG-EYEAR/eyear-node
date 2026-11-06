"""EYEAR Node 메시지 모델 (원본). JSON Schema는 schema/export.py로 이 파일에서 생성한다.

메시지 종류
- NodeStateMessage : Node → 중앙. 방 전체 스냅샷 (HEARTBEAT / STATE_CHANGE)
- SpeakCommand     : 중앙 → Node. 스피커로 말할 문장
- VoiceReply       : Node → 중앙. SpeakCommand 질문에 대한 대답 (인식 텍스트)
- VoiceCommand     : Node → 중앙. "하이 엘지" 뒤 발화 (인식 텍스트)

원본 영상·음성은 어떤 메시지에도 들어가지 않는다 (R-10).
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from schema.enums import (
    PERSON_CHANGES,
    RESERVED_IDENTITIES,
    Activity,
    AudioContext,
    ChangeType,
    EventType,
    IdentitySource,
    Light,
    Motion,
    OccupancyState,
    Posture,
)

SCHEMA_VERSION = "0.1"

Confidence = Annotated[float, Field(ge=0.0, le=1.0, description="신뢰도 0~1")]
Identifier = Annotated[str, Field(pattern=r"^[a-z0-9_]+$", description="소문자, 숫자, 밑줄")]
IdentityName = Annotated[
    str,
    Field(
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="등록된 가족 이름(영문 대문자) 또는 예약어 UNKNOWN / GUEST",
    ),
]


def now() -> datetime:
    return datetime.now().astimezone()


class _Model(BaseModel):
    # 모르는 필드가 오면 거부 → 오타나 버전 불일치를 바로 발견
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class _Envelope(_Model):
    """모든 Node 발신 메시지의 공통 헤더 (R-11)."""

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    event_id: UUID = Field(default_factory=uuid4)
    node_id: Identifier
    room_id: Identifier
    timestamp: AwareDatetime = Field(default_factory=now, description="ISO 8601, 시간대 포함")


# ---------- 방 상태 ----------

class Position(_Model):
    """방 기준 좌표, 미터 단위. 원점·축 방향은 팀원2와 합의 예정."""

    x: float
    y: float


class Person(_Model):
    track_id: int = Field(ge=0, description="Node 안에서 사람을 구분하는 번호. 방을 나가면 재사용될 수 있음")
    identity: IdentityName
    identity_confidence: Confidence
    identity_source: IdentitySource
    posture: Posture
    posture_confidence: Confidence
    motion: Motion
    activity: Activity
    activity_confidence: Confidence
    position: Position | None = Field(description="레이더가 없거나 위치를 모르면 null")

    @field_validator("identity")
    @classmethod
    def _known_identity(cls, v: str, info: ValidationInfo) -> str:
        # context={"family": {...}}로 검증할 때만 가족 목록을 확인 (송신 측에서 사용)
        family = (info.context or {}).get("family")
        if family is not None and v not in RESERVED_IDENTITIES and v not in family:
            raise ValueError(f"등록되지 않은 identity: {v}")
        return v

    @model_validator(mode="after")
    def _unknown_identity_has_unknown_source(self):
        if (self.identity == "UNKNOWN") != (self.identity_source == IdentitySource.UNKNOWN):
            raise ValueError("identity가 UNKNOWN이면 identity_source도 UNKNOWN이어야 함 (그 반대도)")
        return self


class Change(_Model):
    type: ChangeType
    track_id: int | None = Field(default=None, description="사람 단위 변화면 해당 track_id, 방 단위 변화면 null")

    @model_validator(mode="after")
    def _track_id_matches_type(self):
        if (self.type in PERSON_CHANGES) != (self.track_id is not None):
            raise ValueError(f"{self.type}: 사람 단위 변화는 track_id 필요, 방 단위 변화는 track_id 없음")
        return self


class Occupancy(_Model):
    """방 단위 재실 (LD2410C 주). 위치를 놓쳐도 사람이 있으면 OCCUPIED + persons: []."""

    state: OccupancyState
    confidence: Confidence


class Audio(_Model):
    context: AudioContext
    confidence: Confidence


class NodeStateMessage(_Envelope):
    """방 전체 스냅샷. 메시지 하나만 받아도 방 상태를 알 수 있다."""

    seq: int = Field(ge=0, description="Node별 일련번호. 1씩 증가, 빠진 번호 = 유실. Node 재시작 시 0부터")
    event_type: EventType
    changes: list[Change] = Field(default_factory=list, description="HEARTBEAT면 빈 리스트")
    occupancy: Occupancy
    persons: list[Person] = Field(description="위치·신원을 추적 중인 사람들. 방을 나간 사람은 빠짐")
    audio: Audio
    light: Light

    @model_validator(mode="after")
    def _consistent(self):
        if self.event_type == EventType.HEARTBEAT and self.changes:
            raise ValueError("HEARTBEAT에는 changes가 없어야 함")
        if self.event_type == EventType.STATE_CHANGE and not self.changes:
            raise ValueError("STATE_CHANGE에는 changes가 하나 이상 있어야 함")
        track_ids = [p.track_id for p in self.persons]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("persons의 track_id가 중복됨")
        if self.persons and self.occupancy.state == OccupancyState.EMPTY:
            raise ValueError("persons가 있는데 occupancy가 EMPTY")
        return self


# ---------- 음성 ----------

class SpeakCommand(_Model):
    """중앙 → Node. Node는 text를 TTS로 출력하고, expect_reply면 출력이 끝난 뒤 듣는다."""

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    command_id: UUID = Field(default_factory=uuid4)
    timestamp: AwareDatetime = Field(default_factory=now)
    text: str = Field(min_length=1)
    expect_reply: bool
    listen_seconds: float = Field(default=5.0, gt=0, le=10, description="expect_reply일 때 최대 듣기 시간(초)")


class VoiceReply(_Envelope):
    """Node → 중앙. SpeakCommand 질문에 대한 대답. "응/아니" 해석은 중앙이 한다."""

    type: Literal["VOICE_REPLY"] = "VOICE_REPLY"
    command_id: UUID = Field(description="대답한 SpeakCommand의 command_id")
    text: str | None = Field(description="인식한 문장. 응답이 없으면 null")
    confidence: Confidence | None = Field(description="음성 인식 신뢰도. 응답이 없으면 null")
    no_response: bool

    @model_validator(mode="after")
    def _no_response_consistent(self):
        if self.no_response != (self.text is None) or (self.text is None) != (self.confidence is None):
            raise ValueError("no_response면 text·confidence는 null, 아니면 둘 다 값이 있어야 함")
        return self


class VoiceCommand(_Envelope):
    """Node → 중앙. 웨이크워드 "하이 엘지" 뒤의 발화."""

    type: Literal["VOICE_COMMAND"] = "VOICE_COMMAND"
    text: str = Field(min_length=1, description="웨이크워드를 뺀 인식 문장")
    confidence: Confidence
