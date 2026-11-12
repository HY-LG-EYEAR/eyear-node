from fusion.events import EventBuilder, RoomSnapshot, diff_changes
from publisher.fake_sender import DEFAULT_SCENARIO, RandomWalker, load_scenario
from schema.family import load_family

FAMILY = load_family()


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _replay(scenario):
    """시나리오 단계를 차례로 적용하며 방별 changes 목록을 모은다."""
    snapshots, prev, result = {}, {}, []
    for step in scenario.steps:
        snapshots.update(step.rooms)
        step_changes = {}
        for room, snap in snapshots.items():
            changes = [(c.type, c.track_id) for c in diff_changes(prev.get(room), snap)]
            if changes:
                step_changes[room] = changes
            prev[room] = snap
        result.append((step.id, step_changes))
    return dict(result)


def test_scenario_produces_expected_changes():
    changes = _replay(load_scenario(DEFAULT_SCENARIO, FAMILY))
    assert changes["t0"] == {}
    assert changes["t1"] == {"room1": [("PERSON_ENTERED", 1), ("OCCUPANCY_CHANGED", None)]}
    assert changes["t2"] == {"room1": [("POSTURE_CHANGED", 1), ("ACTIVITY_CHANGED", 1)]}
    assert changes["t3"] == {}
    assert changes["t4"] == {"room1": [("IDENTITY_CHANGED", 1), ("POSTURE_CHANGED", 1), ("LIGHT_CHANGED", None)]}  # 어둠 속 자세 UNKNOWN
    assert changes["t5-1"] == {"room1": [("AUDIO_CHANGED", None)]}
    assert changes["t5-2"] == {"room1": [
        ("IDENTITY_CHANGED", 1), ("POSTURE_CHANGED", 1), ("ACTIVITY_CHANGED", 1),
        ("AUDIO_CHANGED", None), ("LIGHT_CHANGED", None),
    ]}
    assert changes["t6"] == {"room1": [("PERSON_LEFT", 1), ("OCCUPANCY_CHANGED", None)]}
    assert changes["t7-1"] == {"living": [("PERSON_ENTERED", 2)]}
    assert changes["t7-2"] == {"living": [("POSTURE_CHANGED", 2)]}


def test_event_builder_heartbeat_and_seq():
    scenario = load_scenario(DEFAULT_SCENARIO, FAMILY)
    clock = FakeClock()
    builder = EventBuilder("node_room1", "room1", FAMILY, heartbeat_interval=1.0, clock=clock)
    empty, entered = scenario.steps[0].rooms["room1"], scenario.steps[1].rooms["room1"]

    first = builder.update(empty)
    assert (first.event_type, first.seq) == ("HEARTBEAT", 0)
    clock.t = 0.5
    assert builder.update(empty) is None          # heartbeat 주기 전
    clock.t = 0.6
    change = builder.update(entered)               # 변화는 즉시
    assert (change.event_type, change.seq) == ("STATE_CHANGE", 1)
    clock.t = 1.5
    assert builder.update(entered) is None         # 마지막 전송(0.6초)부터 주기를 다시 셈
    clock.t = 1.6
    assert builder.update(entered).event_type == "HEARTBEAT"


def test_confidence_only_change_is_not_a_change():
    snap = load_scenario(DEFAULT_SCENARIO, FAMILY).steps[2].rooms["room1"]
    data = snap.model_dump()
    data["persons"][0]["identity_confidence"] = 0.5
    data["persons"][0]["position"] = {"x": 0.0, "y": 0.0}
    assert diff_changes(snap, RoomSnapshot.model_validate(data)) == []


def test_random_walker_stays_valid():
    walker = RandomWalker(["room1", "room2", "living"], FAMILY, interval=0, seed=7)
    builders = {room: EventBuilder(f"node_{room}", room, FAMILY) for room in walker.snapshots}
    for i in range(2000):
        walker.tick(float(i))  # 내부에서 RoomSnapshot 검증
        for room, builder in builders.items():
            builder.update(walker.snapshots[room])  # NodeStateMessage 검증 (가족 목록 포함)
        identities = [
            p.identity for s in walker.snapshots.values() for p in s.persons if p.identity != "UNKNOWN"
        ]
        assert len(identities) == len(set(identities)), "같은 가족이 두 곳에 동시에 있음"
