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


def radar_config(placed=True, door=None) -> RadarConfig:
    inst = InstallationConfig(1.0, 0.0, 90.0, 3.0, 3.5, *(door or (None, None))) if placed else InstallationConfig()
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

    def __init__(self, placed=True, door=None):
        self.clock = Clock()
        self.radar = RadarInput(radar_config(placed, door), clock=self.clock)
        self.state = RoomState(FAMILY, clock=self.clock, radar=self.radar)

    def run(self, seconds, camera=None, light=Light.BRIGHT, r10=TargetState.NONE, move_e=0,
            targets=(), camera_on=True, radar_on=True, gates=()):
        """실제 Node처럼 0.1초마다 값을 넣고 상태를 계산한다 (main.py는 카메라 프레임마다 계산)."""
        snap = None
        for _ in range(round(seconds / 0.1)):
            self.clock.t += 0.1
            if camera_on:
                self.state.set_camera(camera or [], light)
            if radar_on:
                self.radar.on_ld2410(Ld2410Report(r10, 100, move_e, 150, 60, 150, (0,) * len(gates), tuple(gates)))
                self.radar.on_ld2450(list(targets))
            snap = self.state.snapshot()
        return snap


def person(identity="MEMBER_A", posture="LYING", activity="OTHER", activity_conf=0.0):
    return VisionPerson(track_id=7, bbox=(0, 0, 100, 50), identity=identity, identity_confidence=0.9,
                        posture=posture, posture_confidence=0.8, activity=activity,
                        activity_confidence=activity_conf)


BED = [Ld2450Target(500, 2000, 0, 320)]      # 레이더 기준 왼쪽 0.5m, 앞 2.0m (가만히)
WALK_IN = [Ld2450Target(500, 2000, 30, 320)]  # 같은 자리로 걸어 들어오는 중 (진짜 사람은 처음 잡힐 때 움직임)


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
    snap = room.run(1, targets=[Ld2450Target(-1500, 1500, 0, 320)])  # LD2450에만 보이는 가짜 대상 (간섭)
    assert snap.occupancy.state == "EMPTY" and snap.persons == []


def test_wall_target_outside_room_is_dropped():
    room = Room()
    snap = room.run(1, r10=TargetState.STATIONARY, targets=[Ld2450Target(-2500, 1000, 0, 320)])  # x=3.5m: 옆벽(3.0m) 너머
    assert snap.persons == []
    assert (snap.occupancy.state, snap.occupancy.confidence) == ("EMPTY", 0.6)  # 방 안에서 본 사람이 없으니 시작 안 함


def test_scenario_bright_rest_lights_off_get_up_and_leave():
    room = Room()
    room.run(1, camera=[person(posture="STANDING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)  # 들어옴
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
    snap = room.run(2, light=Light.DARK, r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    [p] = snap.persons
    assert (p.identity, p.identity_source, p.posture, p.motion) == ("UNKNOWN", "UNKNOWN", "UNKNOWN", "ACTIVE")
    snap = room.run(2, light=Light.DARK, r10=TargetState.STATIONARY)  # 위치를 놓쳐도 같은 사람으로 유지
    [p2] = snap.persons
    assert (p2.track_id, p2.identity, p2.position) == (p.track_id, "UNKNOWN", None)


def test_ld2410c_alone_does_not_start_occupancy_back_lobe():
    """레이더 뒤쪽(벽 너머 거실) 사람: LD2410C만 "있음", LD2450·카메라는 못 봄 → 시작하지 않음 (2026-11-12 실측)."""
    room = Room()
    snap = room.run(10, r10=TargetState.STATIONARY, move_e=0)
    assert (snap.occupancy.state, snap.occupancy.confidence, snap.persons) == ("EMPTY", 0.6, [])


def test_walking_out_is_empty_even_if_ld2410c_still_present():
    """걸어 나감: LD2450이 움직이던 사람을 놓치고 5초 → EMPTY. 레이더 뒤 사람 때문에 LD2410C가 계속 "있음"이어도."""
    room = Room()
    room.run(3, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=90,
             targets=[Ld2450Target(-800, 1500, 40, 320)])
    snap = room.run(4, r10=TargetState.STATIONARY)   # 나간 직후 (아직 5초 안 됨)
    assert snap.occupancy.state == "OCCUPIED"
    snap = room.run(4, r10=TargetState.STATIONARY)   # 놓친 뒤 "최근 1초" + 5초가 지남 (실제로는 나간 뒤 6~7초)
    assert (snap.occupancy.state, snap.occupancy.confidence, snap.persons) == ("EMPTY", 0.85, [])
    snap = room.run(5, r10=TargetState.STATIONARY)   # 뒤쪽 사람만 남음 → 다시 시작하지 않음
    assert snap.occupancy.state == "EMPTY"


def test_still_person_lost_by_ld2450_is_kept_by_ld2410c():
    """가만히 누운 사람을 LD2450이 놓쳐도(움직이지 않던 중) 나감이 아님 → LD2410C로 유지."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(4, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED)
    snap = room.run(30, light=Light.DARK, r10=TargetState.STATIONARY)
    assert snap.occupancy.state == "OCCUPIED" and snap.persons[0].activity == "RESTING"


def test_entering_does_not_flicker_entered_left_entered():
    """레이더가 먼저 잡고 → 잠깐 놓치고 → 카메라가 잡아도 같은 사람 하나로 이어짐."""
    room = Room()
    ids = []
    for kw in (dict(targets=WALK_IN), dict(), dict(camera=[person(posture="STANDING")])):
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


def test_person_seen_only_by_camera_walking_behind_radar_is_left():
    """2026-11-12 실측: 카메라에만 잠깐 보이던 사람이 움직여 레이더 뒤로 감 → LD2410C 뒤쪽 감지로 "있음"이 남아도 5초 뒤 EMPTY."""
    room = Room()
    room.run(2, camera=[person(posture="STANDING")], r10=TargetState.MOVING, move_e=90)  # LD2450은 못 봄
    snap = room.run(3, r10=TargetState.STATIONARY, move_e=20)
    assert snap.occupancy.state == "OCCUPIED"  # 아직 5초 안 됨 (TRACK_CARRIED)
    snap = room.run(4, r10=TargetState.STATIONARY, move_e=20)
    assert (snap.occupancy.state, snap.persons) == ("EMPTY", [])


def test_lights_off_while_still_is_not_left_even_after_long_time():
    room = Room()
    room.run(4, camera=[person(posture="SITTING")], r10=TargetState.STATIONARY, move_e=5)  # 가만히 앉음 (LD2450 못 봄)
    snap = room.run(60, light=Light.DARK, r10=TargetState.STATIONARY, move_e=5)
    assert snap.occupancy.state == "OCCUPIED" and snap.persons[0].identity_source == "TRACK_CARRIED"


def test_static_ghost_never_moved_is_ignored():
    """2026-11-12 실측: 간섭으로 생긴 움직임 없는 가짜 대상(2.1m) + LD2410C 뒤쪽 감지가 겹쳐도 시작하지 않음."""
    room = Room()
    ghost = [Ld2450Target(-1500, 1500, 0, 320)]
    snap = room.run(10, r10=TargetState.STATIONARY, targets=ghost)
    assert (snap.occupancy.state, snap.persons) == ("EMPTY", [])
    assert room.radar.ignored_static > 0


def test_ghost_does_not_block_walking_out():
    """사람이 나간 뒤 움직임 없는 가짜 대상만 남아도 "나감"."""
    room = Room()
    ghost = Ld2450Target(-1500, 1500, 0, 320)
    room.run(3, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=90,
             targets=[Ld2450Target(-800, 1200, 35, 320), ghost])
    snap = room.run(8, r10=TargetState.STATIONARY, targets=[ghost])
    assert (snap.occupancy.state, snap.persons) == ("EMPTY", [])


def test_still_person_reacquired_at_same_spot_counts():
    """움직이다 앉은 사람을 LD2450이 놓쳤다가 같은 자리에서 다시 잡으면 진짜 사람으로 이어받음."""
    room = Room()
    room.run(2, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(5, camera=[person(posture="SITTING")], r10=TargetState.STATIONARY, move_e=5)   # LD2450 놓침
    snap = room.run(3, camera=[person(posture="SITTING")], r10=TargetState.STATIONARY, move_e=5, targets=BED)
    assert snap.persons[0].position is not None   # 다시 잡은 위치를 씀


def test_someone_passing_by_after_lights_off_does_not_empty_resting_room():
    """2026-11-12 리허설: 누워 쉬는 사람이 있는데, 불을 끄러 레이더 옆을 지나간 사람이 사라지자 EMPTY가 됨 → 고침."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, move_e=0)
    # 불 끄러 온 사람: 레이더 바로 옆에서 움직이다 사라짐, 동시에 불이 꺼져 카메라는 못 봄
    room.run(2, light=Light.DARK, r10=TargetState.MOVING_AND_STATIONARY, move_e=100,
             targets=[Ld2450Target(-140, 220, 25, 320)])
    snap = room.run(10, light=Light.DARK, r10=TargetState.STATIONARY, move_e=0)
    assert snap.occupancy.state == "OCCUPIED"
    assert (snap.persons[0].identity, snap.persons[0].activity) == ("MEMBER_A", "RESTING")


def test_resting_person_who_gets_up_in_dark_and_walks_out_is_empty():
    """어둠 속에서 일어나(ACTIVE 3초 → RESTING 풀림) 걸어 나가면 다시 "나감" 판단."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, move_e=0)
    room.run(5, light=Light.DARK, r10=TargetState.MOVING, move_e=90,
             targets=[Ld2450Target(-600, 1500, 35, 320)])               # 일어나 걸어감
    snap = room.run(8, light=Light.DARK, r10=TargetState.STATIONARY, move_e=10)  # 뒤쪽 사람 때문에 LD2410C는 "있음"
    assert snap.occupancy.state == "EMPTY"


# LD2410C 엔지니어링 모드 칸별 정지 에너지 (칸 0.75m). BED(2.06m) → 침대 칸 round(2.06/0.75) = 3
G_LYING = (0, 0, 4, 15, 40, 70, 30, 20, 10)  # 실측: 누우면 침대 칸 8~25, 3.0m·3.75m 칸은 반사로 늘 높음
G_EMPTY = (0, 0, 4, 4, 25, 40, 20, 10, 10)   # 실측: 빈 방 침대 칸 3~5
G_FULL = (0, 0, 100, 100, 100, 100, 100, 100, 100)  # 다른 사람이 걷는 동안


def _rest_then_toss_then_passerby(room, gates):
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN, gates=gates)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED,
             gates=gates)
    # 어둠 속에서 크게 뒤척임 4초 → RESTING이 풀림 (기존 규칙으로는 "일어났을 수 있음")
    room.run(4, light=Light.DARK, r10=TargetState.MOVING_AND_STATIONARY, move_e=90, targets=BED,
             gates=gates and G_LYING)
    # 다른 사람이 레이더 옆을 지나가다 나감
    room.run(2, light=Light.DARK, r10=TargetState.MOVING_AND_STATIONARY, move_e=100,
             targets=[Ld2450Target(-140, 220, 25, 320)], gates=gates and G_FULL)


def test_bed_gate_keeps_room_occupied_after_passerby_even_if_rest_released():
    """2026-11-19 실측 반영: 지나간 사람이 나간 뒤 침대 칸 정지 에너지가 남아 있으면 누운 사람은 그대로."""
    room = Room()
    _rest_then_toss_then_passerby(room, G_LYING)
    snap = room.run(10, light=Light.DARK, r10=TargetState.STATIONARY, move_e=0, gates=G_LYING)
    assert snap.occupancy.state == "OCCUPIED"
    assert snap.persons[0].identity == "MEMBER_A"

    # 비교: 칸별 값이 없으면(엔지니어링 모드 꺼짐) 같은 상황에서 기존 규칙대로 EMPTY
    room = Room()
    _rest_then_toss_then_passerby(room, ())
    assert room.run(10, light=Light.DARK, r10=TargetState.STATIONARY, move_e=0).occupancy.state == "EMPTY"


def test_bed_gate_empty_lets_walked_out_person_leave():
    """누운 사람이 정말 일어나 나갔으면 침대 칸이 비어 EMPTY (실측: 나가고 몇 초 안에 침대 칸 떨어짐)."""
    room = Room()
    _rest_then_toss_then_passerby(room, G_LYING)
    snap = room.run(8, light=Light.DARK, r10=TargetState.STATIONARY, move_e=0, gates=G_EMPTY)
    assert snap.occupancy.state == "EMPTY"


def test_getting_up_in_light_and_leaving_is_empty_once_bed_gate_clears():
    """불을 켜고 일어나 나가면, 침대 칸이 비는 대로 EMPTY (실측: 나가고 몇 초 안에 침대 칸이 떨어짐)."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN, gates=G_LYING)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED,
             gates=G_LYING)
    room.run(3, camera=[person(posture="STANDING")], r10=TargetState.MOVING, move_e=80,
             targets=[Ld2450Target(-600, 1500, 35, 320)], gates=G_EMPTY)
    snap = room.run(12, r10=TargetState.STATIONARY, move_e=0, gates=G_EMPTY)  # LD2410C "있음"(뒤쪽일 수 있음)이어도
    assert snap.occupancy.state == "EMPTY"


def test_radar_summary_reports_gate_medians_and_target_range():
    room = Room()
    room.run(1, r10=TargetState.STATIONARY, targets=WALK_IN, gates=G_LYING)
    s = room.radar.summary()
    assert s.still_gates == G_LYING
    assert s.target_range_m == pytest.approx(2.06, abs=0.01)


def test_ld2450_keeps_lying_person_when_ld2410c_misses_them_in_dark():
    """2026-11-19 시연 공간: 3.2m에 누운 사람을 LD2410C는 대부분 "없음", LD2450은 계속 잡음 → 유지."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED)
    snap = room.run(20, light=Light.DARK, r10=TargetState.NONE, targets=BED)  # 불 꺼짐, LD2410C "없음"
    assert (snap.occupancy.state, snap.occupancy.confidence) == ("OCCUPIED", 0.7)
    assert (snap.persons[0].identity, snap.persons[0].activity) == ("MEMBER_A", "RESTING")
    # LD2450이 잠깐(1초) 놓쳐도 깜빡이지 않음
    snap = room.run(1, light=Light.DARK, r10=TargetState.NONE)
    assert snap.occupancy.state == "OCCUPIED"
    snap = room.run(2, light=Light.DARK, r10=TargetState.NONE, targets=BED)
    assert snap.occupancy.state == "OCCUPIED"
    # 아무 센서도 못 보는 채로 오래 지나면 비어 있음 (지어내지 않음)
    assert room.run(5, light=Light.DARK, r10=TargetState.NONE).occupancy.state == "EMPTY"


def test_ld2450_alone_still_cannot_start_occupancy():
    """LD2450 혼자 보이는 대상은 여전히 재실을 시작하지 않음 (가짜 대상 대비)."""
    room = Room()
    snap = room.run(5, light=Light.DARK, r10=TargetState.NONE, targets=WALK_IN)
    assert snap.occupancy.state == "EMPTY"


def test_bed_gate_keeps_lying_person_when_both_radars_lose_them():
    """2026-11-19 실측: 지나간 사람이 나간 뒤 LD2450은 누운 사람을 못 잡고, 30초쯤 지나면 LD2410C도 "없음".
    침대 칸 정지 에너지가 남아 있으면 OCCUPIED 유지 (자는 방에 청소기가 들어가지 않게)."""
    room = Room()
    _rest_then_toss_then_passerby(room, G_LYING)
    snap = room.run(30, light=Light.DARK, r10=TargetState.NONE, gates=G_LYING)
    assert (snap.occupancy.state, snap.occupancy.confidence) == ("OCCUPIED", 0.7)
    assert snap.persons[0].identity == "MEMBER_A"
    # 침대 칸이 비면(정말 나감) EMPTY
    assert room.run(12, light=Light.DARK, r10=TargetState.NONE, gates=G_EMPTY).occupancy.state == "EMPTY"


def test_camera_seeing_only_passerby_in_dim_light_keeps_bed_memory():
    """어둑할 때 카메라가 지나가는 사람(서 있음)만 보고 누운 사람은 못 봐도, 침대 칸이 차 있으면 침대 기억 유지."""
    room = Room()
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN, gates=G_LYING)
    room.run(12, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED,
             gates=G_LYING)
    room.run(3, light=Light.DIM, camera=[person(identity="UNKNOWN", posture="WALKING")], r10=TargetState.MOVING,
             move_e=100, targets=[Ld2450Target(-300, 900, 40, 320)], gates=G_FULL)
    snap = room.run(15, light=Light.DARK, r10=TargetState.NONE, gates=G_LYING)
    assert snap.occupancy.state == "OCCUPIED"


# 문이 있는 방: 문 안쪽 (2.5, 0.8) = 레이더 기준 오른쪽 1.5m·앞 0.8m (x_mm -1500)
DOOR = (2.5, 0.8)


def _walk(room, path, r10=TargetState.MOVING_AND_STATIONARY):
    for x, y in path:
        room.run(0.5, light=Light.DARK, r10=r10, move_e=100, targets=[Ld2450Target(x, y, 30, 320)])


def _lie_down_then_dark(room):
    room.run(1, camera=[person(posture="WALKING")], r10=TargetState.MOVING, move_e=80, targets=WALK_IN)
    room.run(6, camera=[person(activity="RESTING", activity_conf=0.8)], r10=TargetState.STATIONARY, targets=BED)


def test_lying_person_kept_until_exit_trace_and_passerby_is_ignored():
    """2026-11-19 실측: 1분 가만히 누우면 모든 센서가 놓침 → 나간 흔적이 없으면 OCCUPIED 0.6 유지.
    문 근처에서 나타났다 문으로 나간 사람(지나감)은 나간 흔적이 아님. 침대에서 시작해 문에서 사라지면 나감."""
    room = Room(door=DOOR)
    _lie_down_then_dark(room)
    _walk(room, [(-1500, 800), (-1200, 1100), (-900, 1400), (-900, 1400), (-1200, 1100), (-1500, 800)])  # 지나감
    snap = room.run(60, light=Light.DARK, r10=TargetState.NONE)   # 모든 센서가 못 봄
    assert (snap.occupancy.state, snap.occupancy.confidence) == ("OCCUPIED", 0.6)
    assert (snap.persons[0].identity, snap.persons[0].identity_source) == ("MEMBER_A", "TRACK_CARRIED")

    _walk(room, [(500, 2000), (0, 1700), (-600, 1300), (-1100, 1000), (-1400, 850)], r10=TargetState.MOVING)  # 일어나 나감
    snap = room.run(8, light=Light.DARK, r10=TargetState.STATIONARY)   # LD2410C "있음"은 뒤쪽일 수 있음
    assert snap.occupancy.state == "EMPTY"


def test_bright_room_with_nobody_on_camera_forgets_lying_person():
    """밝은데 카메라가 10초 동안 아무도 못 보면 누워 있던 기억을 지움 (밝으면 누운 사람도 카메라가 봄)."""
    room = Room(door=DOOR)
    _lie_down_then_dark(room)
    assert room.run(5, r10=TargetState.NONE).occupancy.state == "OCCUPIED"
    assert room.run(7, r10=TargetState.NONE).occupancy.state == "EMPTY"


# ---------- 목소리 신원 (화자 식별) ----------

def test_voice_identity_in_dark_replaces_track_carried():
    """t5: 어둠 속 "하이 엘지, 불 켜줘"를 목소리로 A라고 알면 identity_source VOICE. 시간이 지나면 TRACK_CARRIED로 이어감."""
    room = Room()
    room.run(1, camera=[person(identity="MEMBER_D", posture="WALKING")], r10=TargetState.MOVING, move_e=80,
             targets=WALK_IN)
    room.run(6, camera=[person(identity="MEMBER_D", activity="RESTING", activity_conf=0.8)],
             r10=TargetState.STATIONARY, targets=BED)
    snap = room.run(5, light=Light.DARK, r10=TargetState.STATIONARY)
    assert snap.persons[0].identity_source == "TRACK_CARRIED"
    room.state.note_voice("MEMBER_D", 0.84)
    snap = room.run(1, light=Light.DARK, r10=TargetState.STATIONARY)
    p = snap.persons[0]
    assert (p.identity, p.identity_source, p.identity_confidence) == ("MEMBER_D", "VOICE", 0.84)
    snap = room.run(31, light=Light.DARK, r10=TargetState.STATIONARY)   # 30초 지나면
    p = snap.persons[0]
    assert (p.identity, p.identity_source) == ("MEMBER_D", "TRACK_CARRIED")
    assert p.identity_confidence == pytest.approx(0.84 * 0.8)


def test_voice_identifies_person_who_entered_dark_room():
    """처음부터 어두운 방에 들어온 사람은 UNKNOWN이지만, 말하면 목소리로 신원을 앎."""
    room = Room()
    snap = room.run(3, light=Light.DARK, camera=[], r10=TargetState.MOVING, move_e=70, targets=WALK_IN)
    assert snap.persons[0].identity == "UNKNOWN"
    room.state.note_voice("MEMBER_B", 0.77)
    p = room.run(1, light=Light.DARK, camera=[], r10=TargetState.STATIONARY, targets=BED).persons[0]
    assert (p.identity, p.identity_source) == ("MEMBER_B", "VOICE")


def test_face_wins_over_voice():
    room = Room()
    room.run(1, camera=[person(identity="MEMBER_D", posture="STANDING")], r10=TargetState.MOVING, move_e=80,
             targets=WALK_IN)
    room.state.note_voice("MEMBER_B", 0.7)
    p = room.run(1, camera=[person(identity="MEMBER_D", posture="STANDING")], r10=TargetState.STATIONARY,
                 targets=BED).persons[0]
    assert (p.identity, p.identity_source) == ("MEMBER_D", "FACE")


def test_unrecognized_face_in_dim_light_uses_voice():
    room = Room()
    room.state.note_voice("MEMBER_D", 0.8)
    p = room.run(2, light=Light.DIM, camera=[person(identity="UNKNOWN", posture="SITTING")],
                 r10=TargetState.MOVING, move_e=70, targets=WALK_IN).persons[0]
    assert (p.identity, p.identity_source) == ("MEMBER_D", "VOICE")
