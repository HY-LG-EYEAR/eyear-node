"""1인 방 연속성: 방에 한 명만 있다고 가정하고, 카메라 트랙이 끊겨도 바깥으로는 같은 사람으로 이어 보낸다.

2026-11-10 실측: 누워 있는 중 잘못된 상자 하나로 카메라 트랙 번호가 바뀌면 RESTING이 처음부터 다시 시작돼
중앙이 질문을 한 번 더 할 수 있었고, 같은 사람이 0.5~1초씩 중복으로 잡혀 PERSON_ENTERED/LEFT가 나갔다.
시나리오에서 방1에는 한 명만 있다 (CLAUDE.md "방에 1명일 때만 같은 사람으로 이어 간다").

- 내보내는 사람은 최대 1명. 여럿이 잡히면 지금 따라가는 카메라 트랙 → 같은 가족 → 가장 큰 상자 순으로 하나만 고른다
- 번호(track_id)는 Node가 정하는 고정 번호. 카메라 트랙이 바뀌어도 그대로
- 신원: 새로 고른 카메라 트랙이 아직 UNKNOWN이면 이어 오던 신원을 유지. 다른 가족으로 확정되면 바꾼다
- 카메라가 잠깐 놓치면(이불 등) hold_seconds 동안 마지막 상태 유지. 그 뒤에도 없으면 비우고,
  다음에 나타나는 사람은 새 번호 (나갔다 들어온 것)
- 여러 명이 있을 수 있는 방(거실)에서는 config로 끈다
"""

from dataclasses import replace

from schema.enums import IDENTITY_UNKNOWN
from vision.tracker import VisionPerson


def _area(p: VisionPerson) -> float:
    return (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])


class SinglePersonContinuity:
    def __init__(self, hold_seconds: float = 3.0):
        self.hold_seconds = hold_seconds
        self._id = 0                       # 지금 내보내는 사람의 번호 (0 = 아직 없음)
        self._camera_track: int | None = None
        self._identity: tuple[str, float] | None = None
        self._last: VisionPerson | None = None
        self._last_seen = -float("inf")

    def update(self, persons: list[VisionPerson], t: float) -> list[VisionPerson]:
        if not persons:
            if self._last is not None and t - self._last_seen <= self.hold_seconds:
                return [self._last]  # 잠깐 놓침: 마지막 상태 유지
            self._last, self._camera_track, self._identity = None, None, None
            return []

        chosen = self._choose(persons)
        if self._last is None:
            self._id += 1  # 새로 나타난 사람
        self._camera_track = chosen.track_id
        if chosen.identity != IDENTITY_UNKNOWN:
            self._identity = (chosen.identity, chosen.identity_confidence)
        identity, confidence = self._identity or (IDENTITY_UNKNOWN, 0.0)
        self._last = replace(chosen, track_id=self._id, identity=identity, identity_confidence=confidence)
        self._last_seen = t
        return [self._last]

    def _choose(self, persons: list[VisionPerson]) -> VisionPerson:
        same_track = [p for p in persons if p.track_id == self._camera_track]
        if same_track:
            return same_track[0]
        if self._identity is not None:
            same_person = [p for p in persons if p.identity == self._identity[0]]
            if same_person:
                return same_person[0]
        return max(persons, key=_area)
