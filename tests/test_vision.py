"""카메라·모델 없이 확인하는 vision 규칙 테스트."""

import numpy as np
import pytest

from fusion.camera_only import snapshot_from_camera
from schema.enums import Light, Posture
from schema.family import load_family
from vision.config import VisionConfig, load_vision_config
from vision.face_id import FaceGallery, FaceMatch, match_embedding
from vision.light import classify_light
from vision.pose import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
    PoseResult, classify_posture,
)
from vision.tracker import CameraTracker

CFG = VisionConfig.default()
FAMILY = load_family()


def make_pose(joints: dict[int, tuple[float, float]]) -> PoseResult:
    points = np.zeros((33, 2), dtype=np.float32)
    visibility = np.zeros(33, dtype=np.float32)
    for index, xy in joints.items():
        points[index] = xy
        visibility[index] = 0.9
    seen = points[visibility > 0]
    x1, y1 = seen.min(axis=0)
    x2, y2 = seen.max(axis=0)
    return PoseResult(points, visibility, (float(x1), float(y1), float(x2), float(y2)))


def body(shoulder, hip, knee=None, ankle=None, nose=None) -> PoseResult:
    """좌우 관절을 가로로 40px씩 벌린 단순한 몸 (좌우 중점이 주어진 위치)."""
    def pair(left, right, xy):
        return {left: (xy[0] - 40, xy[1]), right: (xy[0] + 40, xy[1])}
    joints = pair(L_SHOULDER, R_SHOULDER, shoulder) | pair(L_HIP, R_HIP, hip)
    if knee:
        joints |= pair(L_KNEE, R_KNEE, knee)
    if ankle:
        joints |= pair(L_ANKLE, R_ANKLE, ankle)
    if nose:
        joints[NOSE] = nose
    return make_pose(joints)


STANDING = body((300, 100), (300, 250), (300, 400), (300, 550), nose=(300, 60))
SITTING = body((300, 100), (300, 250), (420, 260), (420, 400), nose=(300, 60))
LYING = body((100, 300), (260, 310), (420, 315), (580, 320), nose=(60, 295))


@pytest.mark.parametrize(
    "pose, expected",
    [
        (STANDING, Posture.STANDING),
        (SITTING, Posture.SITTING),
        (LYING, Posture.LYING),
        (body((300, 100), (300, 250), (300, 400)), Posture.STANDING),   # 발목 안 보임: 허벅지 기울기로 판단
        (body((300, 100), (300, 250), (420, 260)), Posture.SITTING),
        (body((300, 100), (300, 250)), Posture.UNKNOWN),                # 다리가 안 보임 (책상 앞)
        (body((300, 100), (380, 220)), Posture.UNKNOWN),                # 눕는 중간 (몸통 약 34~60도)
        (make_pose({L_SHOULDER: (300, 100)}), Posture.UNKNOWN),         # 엉덩이 안 보임
    ],
)
def test_classify_posture(pose, expected):
    judgement = classify_posture(pose, CFG.pose)
    assert judgement.posture == expected
    assert (judgement.confidence > 0) == (expected != Posture.UNKNOWN)


def test_light_levels():
    def gray(v):
        return np.full((90, 160, 3), v, dtype=np.uint8)
    assert classify_light(gray(10), CFG.light)[0] == Light.DARK
    assert classify_light(gray(70), CFG.light)[0] == Light.DIM
    assert classify_light(gray(200), CFG.light)[0] == Light.BRIGHT
    assert classify_light(None, CFG.light) == (Light.UNKNOWN, None)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _gallery():
    rng = np.random.default_rng(0)
    a, b = _unit(rng.normal(size=512)), _unit(rng.normal(size=512))
    return FaceGallery({"MEMBER_A": np.stack([a]), "MEMBER_B": np.stack([b])}), a, b


def test_match_embedding():
    gallery, a, b = _gallery()
    noise = _unit(np.random.default_rng(1).normal(size=512))
    assert match_embedding(gallery, _unit(a + 0.3 * noise), CFG.face)[0] == "MEMBER_A"
    assert match_embedding(gallery, noise, CFG.face)[0] == "UNKNOWN"           # 등록 안 된 얼굴
    assert match_embedding(gallery, _unit(a + b), CFG.face)[0] == "UNKNOWN"     # 두 사람과 똑같이 닮음
    assert match_embedding(FaceGallery({}), a, CFG.face) == ("UNKNOWN", 0.0, 0.0)


def _face_on(pose: PoseResult, identity: str, confidence: float = 0.8) -> FaceMatch:
    nx, ny = pose.points[NOSE]
    return FaceMatch((nx - 30, ny - 40, nx + 30, ny + 40), identity, 0.6, confidence)


def test_tracker_identity_vote_and_sticky():
    tracker = CameraTracker(CFG.face, CFG.pose, CFG.tracking)
    t = 0.0
    for _ in range(2):  # 투표 2번은 아직 부족 (vote_min=3)
        persons = tracker.update([_face_on(STANDING, "MEMBER_A")], [STANDING], t)
        t += 0.1
    assert persons[0].identity == "UNKNOWN"
    persons = tracker.update([_face_on(STANDING, "MEMBER_A")], [STANDING], t)
    assert (persons[0].identity, persons[0].track_id) == ("MEMBER_A", 1)
    # 고개를 돌려 얼굴이 안 보여도 신원 유지
    for _ in range(5):
        t += 0.1
        persons = tracker.update([], [STANDING], t)
    assert persons[0].identity == "MEMBER_A"


def test_tracker_posture_needs_hold_and_walking():
    tracker = CameraTracker(CFG.face, CFG.pose, CFG.tracking)
    tracker.update([], [STANDING], 0.0)
    assert tracker.update([], [STANDING], 0.3)[0].posture == Posture.UNKNOWN  # 0.5초 유지 전
    assert tracker.update([], [STANDING], 0.6)[0].posture == Posture.STANDING
    # 몸 중심이 옆으로 빠르게 이동 → WALKING
    t, x = 0.6, 0.0
    for _ in range(12):
        t += 0.1
        x += 30  # 0.1초에 30px = 몸통 길이(150px)의 2배/초
        moved = make_pose({i: (p[0] + x, p[1]) for i, p in enumerate(STANDING.points) if STANDING.visibility[i] > 0})
        persons = tracker.update([], [moved], t)
    assert persons[0].posture == Posture.WALKING


def test_tracker_drops_lost_person():
    tracker = CameraTracker(CFG.face, CFG.pose, CFG.tracking)
    tracker.update([], [STANDING], 0.0)
    assert len(tracker.update([], [], 0.5)) == 1
    assert tracker.update([], [], 1.6) == []


def test_camera_only_snapshot_does_not_invent_values():
    tracker = CameraTracker(CFG.face, CFG.pose, CFG.tracking)
    persons = tracker.update([], [STANDING], 0.0)
    snap = snapshot_from_camera(persons, Light.BRIGHT, FAMILY)
    p = snap.persons[0]
    assert snap.occupancy.state == "OCCUPIED"
    assert (p.identity, p.identity_source) == ("UNKNOWN", "UNKNOWN")
    assert (p.motion, p.activity, p.position) == ("UNKNOWN", "UNKNOWN", None)
    assert snap.audio.context == "UNKNOWN"
    empty = snapshot_from_camera([], Light.BRIGHT, FAMILY)
    assert (empty.occupancy.state, empty.occupancy.confidence) == ("EMPTY", 0.5)
    broken = snapshot_from_camera(None, Light.UNKNOWN, FAMILY)
    assert (broken.occupancy.state, broken.light) == ("UNKNOWN", "UNKNOWN")


def test_config_file_matches_defaults_keys():
    load_vision_config()  # 알 수 없는 키가 있으면 실패
