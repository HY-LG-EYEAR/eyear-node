"""방 상태 스냅샷 → NodeStateMessage 생성.

이전 스냅샷과 비교해 changes를 만들고, 변화가 없으면 heartbeat 주기마다 HEARTBEAT를 낸다.
가짜 송신기와 실제 Node가 같은 코드를 쓴다.
"""

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from schema.enums import ChangeType, EventType, Light
from schema.models import Audio, Change, NodeStateMessage, Occupancy, Person


class RoomSnapshot(BaseModel):
    """메시지 헤더를 뺀 방 상태."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    occupancy: Occupancy
    persons: list[Person]
    audio: Audio
    light: Light


def diff_changes(prev: RoomSnapshot | None, curr: RoomSnapshot) -> list[Change]:
    """신뢰도·위치·motion만 바뀐 것은 변화로 보지 않는다 (heartbeat에 담김)."""
    if prev is None:
        return []
    changes: list[Change] = []
    before = {p.track_id: p for p in prev.persons}
    after = {p.track_id: p for p in curr.persons}

    for track_id in sorted(before.keys() - after.keys()):
        changes.append(Change(type=ChangeType.PERSON_LEFT, track_id=track_id))
    for track_id in sorted(after.keys() - before.keys()):
        changes.append(Change(type=ChangeType.PERSON_ENTERED, track_id=track_id))
    for track_id in sorted(before.keys() & after.keys()):
        b, a = before[track_id], after[track_id]
        if (b.identity, b.identity_source) != (a.identity, a.identity_source):
            changes.append(Change(type=ChangeType.IDENTITY_CHANGED, track_id=track_id))
        if b.posture != a.posture:
            changes.append(Change(type=ChangeType.POSTURE_CHANGED, track_id=track_id))
        if b.activity != a.activity:
            changes.append(Change(type=ChangeType.ACTIVITY_CHANGED, track_id=track_id))

    if prev.occupancy.state != curr.occupancy.state:
        changes.append(Change(type=ChangeType.OCCUPANCY_CHANGED))
    if prev.audio.context != curr.audio.context:
        changes.append(Change(type=ChangeType.AUDIO_CHANGED))
    if prev.light != curr.light:
        changes.append(Change(type=ChangeType.LIGHT_CHANGED))
    return changes


class EventBuilder:
    """Node 하나의 메시지 생성기. update()를 자주(예: 0.1초마다) 부르면 된다.

    첫 메시지는 HEARTBEAT(비교할 이전 상태가 없음)이며, 그 자체로 방 전체 스냅샷이다.
    """

    def __init__(
        self,
        node_id: str,
        room_id: str,
        family: frozenset[str] | set[str],
        heartbeat_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.node_id = node_id
        self.room_id = room_id
        self.family = family
        self.heartbeat_interval = heartbeat_interval
        self._clock = clock
        self._seq = 0
        self._prev: RoomSnapshot | None = None
        self._last_sent: float | None = None

    def update(self, snapshot: RoomSnapshot) -> NodeStateMessage | None:
        now = self._clock()
        changes = diff_changes(self._prev, snapshot)
        if changes:
            event_type = EventType.STATE_CHANGE
        elif self._last_sent is None or now - self._last_sent >= self.heartbeat_interval:
            event_type = EventType.HEARTBEAT
        else:
            return None

        message = NodeStateMessage.model_validate(
            {
                "seq": self._seq,
                "node_id": self.node_id,
                "room_id": self.room_id,
                "event_type": event_type,
                "changes": [c.model_dump() for c in changes],
                **snapshot.model_dump(),
            },
            context={"family": self.family},
        )
        self._seq += 1
        self._prev = snapshot
        self._last_sent = now
        return message
