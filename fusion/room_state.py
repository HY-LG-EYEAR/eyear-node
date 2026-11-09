"""센서별 최신 결과를 모아 방 상태(RoomSnapshot)를 만든다.

카메라·마이크는 각자 다른 속도로 결과를 낸다 (카메라 매 프레임, 소리 1초마다, 서로 다른 스레드).
RoomState가 마지막 결과와 받은 시각을 들고 있다가, snapshot()을 부를 때 합친다.

모르는 값은 지어내지 않는다 (R-09, R-13):
- 센서를 끈 경우(--no-camera 등), 또는 stale_seconds 넘게 결과가 오지 않으면 그 센서 값은 UNKNOWN
  - 카메라: 재실 UNKNOWN, 사람 목록 비움, 밝기 UNKNOWN
  - 마이크: audio UNKNOWN
- 위치(position): null, 움직임(motion): UNKNOWN (레이더 담당, M5)
- 활동(activity): UNKNOWN (RESTING 판단은 M4)
- 카메라에 아무도 안 보이면 EMPTY지만 신뢰도 0.5 (카메라 밖에 사람이 있을 수 있음, 2026-11-06 결정)
"""

import threading
import time
from collections.abc import Callable

from fusion.events import RoomSnapshot
from schema.enums import IDENTITY_UNKNOWN, Light
from schema.models import Audio
from vision.tracker import VisionPerson

SEEN_CONFIDENCE = 0.9
NOT_SEEN_CONFIDENCE = 0.5
UNKNOWN_AUDIO = Audio(context="UNKNOWN", confidence=0.0)


def build_snapshot(
    persons: list[VisionPerson] | None,
    light: Light,
    audio: Audio | None,
    family: frozenset[str],
) -> RoomSnapshot:
    """persons=None이면 카메라를 쓸 수 없는 상태 → 재실 UNKNOWN. audio=None이면 소리 UNKNOWN."""
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
            "audio": (audio or UNKNOWN_AUDIO).model_dump(),
            "light": light,
        },
        context={"family": family},
    )


class RoomState:
    """여러 스레드에서 set_*을 부르고, 메시지를 만드는 쪽에서 snapshot()을 부른다."""

    def __init__(
        self,
        family: frozenset[str],
        stale_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.family = family
        self.stale_seconds = stale_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._camera: tuple[list[VisionPerson], Light] | None = None
        self._camera_at = -float("inf")
        self._audio: Audio | None = None
        self._audio_at = -float("inf")

    def set_camera(self, persons: list[VisionPerson] | None, light: Light) -> None:
        """persons=None: 카메라 프레임을 읽지 못함."""
        with self._lock:
            self._camera = None if persons is None else (list(persons), light)
            self._camera_at = self._clock()

    def set_audio(self, audio: Audio | None) -> None:
        with self._lock:
            self._audio = audio
            self._audio_at = self._clock()

    def snapshot(self) -> RoomSnapshot:
        now = self._clock()
        with self._lock:
            camera = self._camera if now - self._camera_at <= self.stale_seconds else None
            audio = self._audio if now - self._audio_at <= self.stale_seconds else None
        persons, light = camera if camera is not None else (None, Light.UNKNOWN)
        return build_snapshot(persons, light, audio, self.family)
