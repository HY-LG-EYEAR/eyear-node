"""카메라 결과만으로 방 상태 만들기 (레이더·마이크가 붙기 전 임시).

카메라가 모르는 값은 지어내지 않는다 (R-09):
- 위치(position): null (레이더 담당)
- 움직임(motion): UNKNOWN (레이더 담당)
- 활동(activity): UNKNOWN (RESTING 판단은 단계 M4)
- 소리(audio): UNKNOWN (마이크 담당)
- 아무도 안 보이면 EMPTY지만 신뢰도 0.5 (카메라 밖에 사람이 있을 수 있음, 2026-11-06 결정)
"""

from fusion.events import RoomSnapshot
from schema.enums import IDENTITY_UNKNOWN, Light
from vision.tracker import VisionPerson

SEEN_CONFIDENCE = 0.9
NOT_SEEN_CONFIDENCE = 0.5


def snapshot_from_camera(persons: list[VisionPerson] | None, light: Light, family: frozenset[str]) -> RoomSnapshot:
    """persons=None이면 카메라 자체를 쓸 수 없는 상태 → 재실 UNKNOWN."""
    if persons is None:
        occupancy = {"state": "UNKNOWN", "confidence": 0.0}
        persons = []
    elif persons:
        occupancy = {"state": "OCCUPIED", "confidence": SEEN_CONFIDENCE}
    else:
        occupancy = {"state": "EMPTY", "confidence": NOT_SEEN_CONFIDENCE}

    return RoomSnapshot.model_validate(
        {
            "occupancy": occupancy,
            "persons": [
                {
                    "track_id": p.track_id,
                    "identity": p.identity,
                    "identity_confidence": p.identity_confidence,
                    "identity_source": "UNKNOWN" if p.identity == IDENTITY_UNKNOWN else "FACE",
                    "posture": p.posture,
                    "posture_confidence": p.posture_confidence,
                    "motion": "UNKNOWN",
                    "activity": "UNKNOWN",
                    "activity_confidence": 0.0,
                    "position": None,
                }
                for p in persons
            ],
            "audio": {"context": "UNKNOWN", "confidence": 0.0},
            "light": light,
        },
        context={"family": family},
    )
