"""EYEAR Node 메시지에서 쓰는 enum 정의.

모든 관측 enum에는 UNKNOWN이 있다. 센서가 없거나 판단할 수 없으면 UNKNOWN을 보낸다 (R-09, R-13).
"""

from enum import Enum


class EventType(str, Enum):
    HEARTBEAT = "HEARTBEAT"        # 변화 없음, 주기적 스냅샷
    STATE_CHANGE = "STATE_CHANGE"  # 무언가 바뀜, changes에 내용이 있음


class ChangeType(str, Enum):
    # 사람 단위 (track_id 있음)
    PERSON_ENTERED = "PERSON_ENTERED"
    PERSON_LEFT = "PERSON_LEFT"
    IDENTITY_CHANGED = "IDENTITY_CHANGED"  # identity 또는 identity_source가 바뀜 (예: FACE → TRACK_CARRIED)
    POSTURE_CHANGED = "POSTURE_CHANGED"
    ACTIVITY_CHANGED = "ACTIVITY_CHANGED"  # 예: OTHER → RESTING (수면 질문 계기)
    # 방 단위 (track_id 없음)
    OCCUPANCY_CHANGED = "OCCUPANCY_CHANGED"
    AUDIO_CHANGED = "AUDIO_CHANGED"
    LIGHT_CHANGED = "LIGHT_CHANGED"


PERSON_CHANGES = {
    ChangeType.PERSON_ENTERED,
    ChangeType.PERSON_LEFT,
    ChangeType.IDENTITY_CHANGED,
    ChangeType.POSTURE_CHANGED,
    ChangeType.ACTIVITY_CHANGED,
}


class IdentitySource(str, Enum):
    FACE = "FACE"
    VOICE = "VOICE"
    TRACK_CARRIED = "TRACK_CARRIED"  # 얼굴로 식별한 뒤 레이더 추적으로 이어받음
    UNKNOWN = "UNKNOWN"              # 신원을 모름 (identity가 UNKNOWN일 때)


class Posture(str, Enum):
    STANDING = "STANDING"
    SITTING = "SITTING"
    LYING = "LYING"
    WALKING = "WALKING"
    UNKNOWN = "UNKNOWN"


class Motion(str, Enum):
    """레이더 기준 움직임 정도."""
    STILL = "STILL"
    SLIGHT = "SLIGHT"
    ACTIVE = "ACTIVE"
    UNKNOWN = "UNKNOWN"


class Activity(str, Enum):
    RESTING = "RESTING"  # 휴식 (잠과 휴식은 구분하지 않음)
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class OccupancyState(str, Enum):
    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"  # 재실 센서 없음/고장. 빈 방으로 지어내지 않음


class AudioContext(str, Enum):
    SILENT = "SILENT"
    SPEECH = "SPEECH"
    CONVERSATION = "CONVERSATION"
    MUSIC = "MUSIC"
    TV = "TV"
    LOUD_NOISE = "LOUD_NOISE"
    UNKNOWN = "UNKNOWN"


class Light(str, Enum):
    DARK = "DARK"
    DIM = "DIM"
    BRIGHT = "BRIGHT"
    UNKNOWN = "UNKNOWN"


# identity 예약어. 가족 이름으로 쓸 수 없다.
IDENTITY_UNKNOWN = "UNKNOWN"  # 판단 불가
IDENTITY_GUEST = "GUEST"      # 등록되지 않은 인물
RESERVED_IDENTITIES = {IDENTITY_UNKNOWN, IDENTITY_GUEST}
