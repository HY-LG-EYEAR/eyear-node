"""카메라 + 레이더 융합 테스트 (가짜 시계, 실물 없이). 시나리오 t1~t6 흐름을 따라간다."""

import pytest

from fusion.radar_input import RadarInput, inside_room, to_room
from fusion.room_state import RoomState
from radar.config import InstallationConfig, MotionConfig, RadarConfig, SensorConfig
from radar.ld2410c import Ld2410Report, TargetState
from radar.ld2450 import Ld2450Target
from schema.enums import Light
from schema.family import load_family
from vision.tracker import VisionPerson

FAMILY = load_family()


def radar_config(placed=True) -> RadarConfig:
    inst = InstallationConfig(1.0, 0.0, 90.0, 3.0, 3.5) if placed else InstallationConfig()
    return RadarConfig(
        ld2450=SensorConfig(port=None, x_positive="left", max_range_m=3.75),
        ld2410c=SensorConfig(port=None),
        installation=inst,       # 방 아래쪽 벽 가운데(1.0, 0)에서 위쪽(+y)을 봄, 방 3.0 × 3.5m
        motion=MotionConfig(),
    )


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Room:
    """시각을 0.1초씩 진행하며 카메라·레이더 값을 넣고 snapshot을 본다."""

    def __init__(self, placed=True):
        self.clock = Clock()
        self.radar = RadarInput(radar_config(placed), clock=self.clock)
        self.state = RoomState(FAMILY, clock=self.clock, radar=self.radar)

    def run(self, seconds, camera=None, light=Light.BRIGHT, r10=TargetState.NONE, move_e=0,
            targets=(), camera_on=True, radar_on=True):
        """실제 Node처럼 0.1초마다 값을 넣고 상태를 계산한다 (main.py는 카메라 프레임마다 계산)."""
        snap = None
        for _ in range(round(seconds / 0.1)):
            self.clock.t += 0.1
            if camera_on:
                self.state.set_camera(camera or [], light)
            if radar_on:
                self.radar.on_ld2410(Ld2410Report(r10, 100, move_e, 150, 60, 150))
                self.radar.on_ld2450(list(targets))
            snap = self.state.snapshot()
        return snap


def person(identity="MEMBER_A", posture="LYING", activity="OTHER", activity_conf=0.0):
    return VisionPerson(track_id=7, bbox=(0, 0, 100, 50), identity=identity, identity_confidence=0.9,
                        posture=posture, posture_confidence=0.8, activity=activity,
                        activity_confidence=activity_conf)


BED = [Ld2450Target(500, 2000, 0, 320)]  # 레이더 기준 왼쪽 0.5m, 앞 2.0m


def test_coordinates_follow_floor_plan_convention():
    cfg = radar_config()
    # 위(+y)를 보는 레이더의 왼쪽 = 평면도 왼쪽(-x)
    assert to_room(Ld2450Target(500, 2000, 0, 0), cfg) == pytest.approx((0.5, 2.0))
    assert to_room(Ld2450Target(-500, 1000, 0, 0), cfg) == pytest.approx((1.5, 1.0))
    assert to_room(Ld2450Target(0, 1000, 0, 0), radar_config(placed=False)) is None  # 설치 정보 없으면 지어내지 않음
    assert not inside_room((1.0, 4.5), cfg)  # 방(3.5m) 밖 = 벽 너머


def test_empty_room_and_ld2450_only_ghost_is_ignored():
    room = Room()
    snap = room.run(1)
    assert (snap.occupancy.state, snap.occupancy.confidence, snap.persons) == ("EMPTY", 0.9, [])
    snap = room.run(1, targets=[Ld2450Target(1500, 1500, 0, 320)])  # LD2450에만 보이는 가짜 대상 (간섭)
    assert snap.occupancy.state == "EMPTY" and snap.persons == []


def test_wall_target_outside_room_is_dropped():
    room = Room()
    snap = room.run(1, r10=TargetState.STATIONARY, targets=[Ld2450Target(-2500, 1000, 0, 320)])  # x=3.5m: 옆벽(3.0m) 너머
    assert snap.persons == []  # LD2410C로 재실은 있지만 위치를 아는 사람은 없음
    assert snap.occupancy.state == "OCCUPIED"


def test_scenario_bright_rest_lights_off_get_up_and_leave():
    room = Room()
    # t1·t2: 밝을 때 누워 쉼 → 카메라 + 레이더
    snap = room.run(4, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED)
    [p] = snap.persons
    assert (snap.occupancy.state, snap.occupancy.confidence) == ("OCCUPIED", 0.95)
    assert (p.identity, p.identity_source, p.posture, p.activity, p.motion) == ("MEMBER_A", "FACE", "LYING", "RESTING", "STILL")
    assert (p.position.x, p.position.y) == (0.5, 2.0)
    first_id = p.track_id

    # t4: 불 꺼짐 → 카메라가 못 봄, LD2450도 놓침 → 레이더(LD2410C)로 이어감
    snap = room.run(10, light=Light.DARK, r10=TargetState.STATIONARY, move_e=5)
    [p] = snap.persons
    assert snap.occupancy.state == "OCCUPIED"
    assert (p.track_id, p.identity, p.identity_source) == (first_id, "MEMBER_A", "TRACK_CARRIED")
    assert p.identity_confidence == pytest.approx(0.72)        # 0.9 × 0.8
    assert (p.posture, p.activity, p.position) == ("UNKNOWN", "RESTING", None)  # 자세는 지어내지 않음

    # 뒤척임(SLIGHT) 정도로는 RESTING 유지
    snap = room.run(5, light=Light.DARK, r10=TargetState.MOVING_AND_STATIONARY, move_e=30)
    assert (snap.persons[0].motion, snap.persons[0].activity) == ("SLIGHT", "RESTING")

    # 크게 움직임(ACTIVE) 3초 이상 → 일어났을 수 있음 → UNKNOWN, 다시 조용해져도 RESTING으로 돌아가지 않음
    snap = room.run(2, light=Light.DARK, r10=TargetState.MOVING, move_e=90)
    assert (snap.persons[0].motion, snap.persons[0].activity) == ("ACTIVE", "RESTING")  # 아직 3초 안 됨
    snap = room.run(3, light=Light.DARK, r10=TargetState.MOVING, move_e=90)
    assert (snap.persons[0].motion, snap.persons[0].activity) == ("ACTIVE", "UNKNOWN")
    snap = room.run(4, light=Light.DARK, r10=TargetState.STATIONARY, move_e=0)
    assert snap.persons[0].activity == "UNKNOWN"

    # t6: 나감 → LD2410C 없음 → EMPTY, 다시 들어오면 새 번호
    snap = room.run(1, light=Light.DARK)
    assert (snap.occupancy.state, snap.persons) == ("EMPTY", [])
    snap = room.run(1, camera=[person(posture="STANDING")], r10=TargetState.MOVING, move_e=80, targets=BED)
    assert snap.persons[0].track_id == first_id + 1


def test_entering_dark_room_identity_unknown():
    room = Room()
    snap = room.run(2, light=Light.DARK, r10=TargetState.MOVING, move_e=80, targets=BED)
    [p] = snap.persons
    assert (p.identity, p.identity_source, p.posture, p.motion) == ("UNKNOWN", "UNKNOWN", "UNKNOWN", "ACTIVE")
    snap = room.run(2, light=Light.DARK, r10=TargetState.STATIONARY)  # 위치를 놓쳐도 같은 사람으로 유지
    [p2] = snap.persons
    assert (p2.track_id, p2.identity, p2.position) == (p.track_id, "UNKNOWN", None)


def test_presence_without_any_person_seen_yet_has_no_person():
    room = Room()
    snap = room.run(2, light=Light.DARK, r10=TargetState.STATIONARY)  # 위치도 신원도 본 적 없음
    assert (snap.occupancy.state, snap.persons) == ("OCCUPIED", [])


def test_entering_does_not_flicker_entered_left_entered():
    """레이더가 먼저 잡고 → 잠깐 놓치고 → 카메라가 잡아도 같은 사람 하나로 이어짐."""
    room = Room()
    ids = []
    for kw in (dict(targets=BED), dict(), dict(camera=[person(posture="STANDING")])):
        snap = room.run(1, r10=TargetState.MOVING, move_e=80, **kw)
        ids.append([p.track_id for p in snap.persons])
    assert ids == [[1], [1], [1]]


def test_stale_radar_falls_back_to_camera_only():
    room = Room()
    room.run(1, r10=TargetState.STATIONARY)
    snap = room.run(3, radar_on=False, light=Light.DARK)  # 레이더 끊김 3초 + 어두움
    assert snap.occupancy.state == "UNKNOWN"  # 지어내지 않음
    snap = room.run(1, radar_on=False, camera=[person(posture="STANDING")])
    assert snap.persons[0].motion == "UNKNOWN" and snap.persons[0].position is None


def test_motion_thresholds_from_measurement():
    room = Room()
    for move_e, expected in ((0, "STILL"), (18, "SLIGHT"), (32, "SLIGHT"), (100, "ACTIVE")):
        snap = room.run(4, camera=[person(posture="SITTING")], r10=TargetState.MOVING_AND_STATIONARY,
                        move_e=move_e, targets=BED)
        assert snap.persons[0].motion == expected, move_e
    fast = [Ld2450Target(500, 2000, 40, 320)]  # 걷기 속도 40cm/s
    snap = room.run(4, camera=[person(posture="STANDING")], r10=TargetState.MOVING, move_e=20, targets=fast)
    assert snap.persons[0].motion == "ACTIVE"
