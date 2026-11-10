"""카메라 화면 속 사람 추적: 얼굴·자세 결과를 사람 단위로 묶고, 여러 프레임에 걸쳐 안정화한다.

- 신원: 최근 얼굴 판단들의 투표. 얼굴이 안 보이는 프레임은 투표하지 않으므로, 고개를 돌려도 신원이 유지된다.
  한 번 확정된 신원은 같은 트랙이 이어지는 동안 유지한다. 얼굴을 못 알아본 판단(UNKNOWN, 누워서 옆얼굴 등)으로는
  지우지 않고, 투표가 다른 가족으로 확정될 때만 바꾼다 (2026-11-09 실측: 누운 동안 45%가 UNKNOWN으로 흔들림).
- 같은 사람 잇기: 이전 상자와 겹치면 같은 사람. 빠르게 걸어 겹치지 않으면 가까운(몸 크기 max_jump배 이내) 사람.
- 한 화면에서 거의 겹치는 두 몸은 한 사람으로 합친다 (이불 등으로 한 사람이 둘로 잡히는 경우).
- 새 사람은 confirm_seconds 이상 보여야 목록에 넣는다 (순간 오검출로 PERSON_ENTERED가 나가지 않게).
- 자세: 새 자세가 hold_seconds 이상 유지돼야 바꾼다 (깜빡임 방지).
- WALKING: 서 있으면서 몸 중심이 빠르게 옆으로 이동.

여기서의 track_id는 카메라 기준 번호다. 레이더와 합치는 것은 fusion(단계 5)에서 한다.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from schema.enums import IDENTITY_UNKNOWN, Activity, Posture
from vision.config import FaceConfig, PoseConfig, TrackingConfig
from vision.face_id import FaceMatch
from vision.pose import NOSE, PoseResult, classify_posture

BBox = tuple[float, float, float, float]


@dataclass
class VisionPerson:
    track_id: int
    bbox: BBox
    identity: str
    identity_confidence: float
    posture: Posture
    posture_confidence: float
    activity: Activity = Activity.UNKNOWN     # vision/activity.py가 채움
    activity_confidence: float = 0.0
    rest_probability: float | None = None     # 최근 CLIP "쉬는 중" 확률 (화면 표시용)
    rest_agree_seconds: float | None = None   # LYING + CLIP 동의가 이어진 시간 (화면 표시용)
    person_probability: float | None = None   # 얼굴 없는 사람의 CLIP "사람" 확률 평균 (화면 표시용)
    face: FaceMatch | None = None   # 이번 프레임의 얼굴 판단 (화면 표시용)
    pose: PoseResult | None = None  # 이번 프레임의 관절 (화면 표시용)


def iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def overlap_of_smaller(a: BBox, b: BBox) -> float:
    """겹친 넓이 / 작은 상자 넓이 (한쪽이 다른 쪽 안에 들어 있으면 1)."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def merge_duplicate_poses(poses: list[PoseResult], overlap: float) -> list[PoseResult]:
    """거의 겹치는 몸은 관절이 더 잘 보이는 쪽 하나만 남긴다."""
    kept: list[PoseResult] = []
    for pose in sorted(poses, key=lambda p: -float(p.visibility.mean())):
        if all(overlap_of_smaller(pose.bbox, k.bbox) < overlap for k in kept):
            kept.append(pose)
    return kept


def _inside(point: tuple[float, float], b: BBox) -> bool:
    return b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]


@dataclass
class _Detection:
    bbox: BBox
    face: FaceMatch | None
    pose: PoseResult | None


def _expand(b: BBox, margin: float) -> BBox:
    mx, my = (b[2] - b[0]) * margin, (b[3] - b[1]) * margin
    return b[0] - mx, b[1] - my, b[2] + mx, b[3] + my


def pair_faces_with_poses(faces: list[FaceMatch], poses: list[PoseResult], margin: float = 0.5) -> list[_Detection]:
    """얼굴을 그 얼굴이 속한 몸에 붙인다. 몸이 없는 얼굴은 얼굴만으로 사람 하나.

    1. 몸의 코 위치가 얼굴 상자 안
    2. 얼굴 중심이 몸 상자 안
    3. 얼굴 중심이 몸 상자를 margin만큼 넓힌 범위 안 (가장 가까운 몸). 누우면 머리가 어깨 바깥이라
       관절로 만든 몸 상자 밖에 얼굴이 놓인다 (2026-11-10 실측).
    """
    detections = [_Detection(p.bbox, None, p) for p in poses]
    for face in faces:
        free = [d for d in detections if d.face is None and d.pose is not None]
        center = _center(face.bbox)
        owner = next((d for d in free if _inside(tuple(d.pose.points[NOSE]), face.bbox)), None)
        if owner is None:
            owner = next((d for d in free if _inside(center, d.bbox)), None)
        if owner is None:
            near = [d for d in free if _inside(center, _expand(d.bbox, margin))]
            owner = min(near, key=lambda d: math.dist(center, _center(d.bbox)), default=None)
        if owner is not None:
            owner.face = face
        else:
            detections.append(_Detection(face.bbox, face, None))
    return detections


@dataclass
class _Track:
    id: int
    bbox: BBox
    last_seen: float
    first_seen: float = 0.0
    has_body: bool = False                               # 몸(관절)이 잡힌 적 있음
    confirmed: tuple[str, float] | None = None           # 확정된 신원 (같은 트랙 동안 유지)
    votes: deque = field(default_factory=deque)          # (identity, confidence)
    posture: Posture = Posture.UNKNOWN
    posture_confidence: float = 0.0
    candidate: Posture = Posture.UNKNOWN
    candidate_since: float = 0.0
    candidate_confidences: list = field(default_factory=list)
    centers: deque = field(default_factory=deque)        # (t, x, torso_length)


class CameraTracker:
    def __init__(self, face: FaceConfig, pose: PoseConfig, tracking: TrackingConfig):
        self.face_cfg = face
        self.pose_cfg = pose
        self.track_cfg = tracking
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, faces: list[FaceMatch], poses: list[PoseResult], t: float) -> list[VisionPerson]:
        poses = merge_duplicate_poses(poses, self.track_cfg.duplicate_overlap)
        detections = pair_faces_with_poses(faces, poses, self.track_cfg.face_body_margin)
        assigned = self._assign(detections, t)
        self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self.track_cfg.lost_seconds]

        persons = []
        for track in sorted(self._tracks, key=lambda tr: tr.id):
            if track.last_seen - track.first_seen < self.track_cfg.confirm_seconds:
                continue  # 아직 확인 중인 새 사람 (실제로 보인 기간 기준)
            detection = assigned.get(track.id)
            identity, identity_conf = self._identity(track)
            persons.append(VisionPerson(
                track_id=track.id,
                bbox=track.bbox,
                identity=identity,
                identity_confidence=identity_conf,
                posture=track.posture,
                posture_confidence=track.posture_confidence,
                face=detection.face if detection else None,
                pose=detection.pose if detection else None,
            ))
        return persons

    def _assign(self, detections: list[_Detection], t: float) -> dict[int, _Detection]:
        """IoU가 큰 쌍부터 이전 사람과 연결, 남은 것은 새 사람."""
        pairs = sorted(
            ((iou(tr.bbox, d.bbox), ti, di) for ti, tr in enumerate(self._tracks) for di, d in enumerate(detections)),
            reverse=True,
        )
        used_tracks, used_dets, assigned = set(), set(), {}
        for score, ti, di in pairs:
            if score < self.track_cfg.iou_match:
                break
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assigned[self._tracks[ti].id] = detections[di]
        # 겹치지 않는 경우 (빠르게 걸음): 가까운 순서로, 직전 상자 크기의 max_jump배 이내면 같은 사람
        near = sorted(
            (self._jump(tr, d), ti, di)
            for ti, tr in enumerate(self._tracks) if ti not in used_tracks
            for di, d in enumerate(detections) if di not in used_dets
        )
        for jump, ti, di in near:
            if jump > self.track_cfg.max_jump:
                break
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assigned[self._tracks[ti].id] = detections[di]
        for di, d in enumerate(detections):
            if di not in used_dets:
                track = _Track(self._next_id, d.bbox, t, first_seen=t)
                self._next_id += 1
                self._tracks.append(track)
                assigned[track.id] = d
        for track in self._tracks:
            if track.id in assigned:
                self._observe(track, assigned[track.id], t)
        return assigned

    def _observe(self, track: _Track, d: _Detection, t: float) -> None:
        # 몸이 있던 사람이 잠깐 얼굴만 잡히면(이불 등) 상자를 얼굴 크기로 줄이지 않는다.
        # 줄이면 몸이 다시 잡혔을 때 겹치지 않아 새 사람으로 처리됨 (2026-11-10 실측)
        if d.pose is not None or not track.has_body:
            track.bbox = d.bbox
        track.has_body = track.has_body or d.pose is not None
        track.last_seen = t
        if d.face is not None and d.face.measured:
            track.votes.append((d.face.identity, d.face.confidence))
            while len(track.votes) > self.face_cfg.vote_window:
                track.votes.popleft()
        if d.pose is not None:
            judgement = classify_posture(d.pose, self.pose_cfg)
            posture = judgement.posture
            if judgement.torso_center is not None:
                track.centers.append((t, judgement.torso_center[0], judgement.torso_length))
                while track.centers and t - track.centers[0][0] > 1.0:
                    track.centers.popleft()
            if posture == Posture.STANDING and self._speed(track) >= self.pose_cfg.walking_speed:
                posture = Posture.WALKING
            self._smooth_posture(track, posture, judgement.confidence, t)

    @staticmethod
    def _jump(track: _Track, d: _Detection) -> float:
        """중심 이동 거리 / 직전 상자의 긴 변."""
        (tx, ty), (dx, dy) = _center(track.bbox), _center(d.bbox)
        size = max(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1], 1.0)
        return ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 / size

    def _speed(self, track: _Track) -> float:
        """최근 1초간 몸 중심의 가로 이동 속도 (몸통 길이/초)."""
        if len(track.centers) < 2:
            return 0.0
        (t0, x0, _), (t1, x1, _) = track.centers[0], track.centers[-1]
        if t1 - t0 < 0.5:
            return 0.0
        torso = sorted(c[2] for c in track.centers)[len(track.centers) // 2]
        return abs(x1 - x0) / (t1 - t0) / max(torso, 1.0)

    def _smooth_posture(self, track: _Track, posture: Posture, confidence: float, t: float) -> None:
        if posture == track.posture:
            track.candidate = posture
            track.posture_confidence = round(0.8 * track.posture_confidence + 0.2 * confidence, 3)
            return
        if posture != track.candidate:
            track.candidate, track.candidate_since, track.candidate_confidences = posture, t, []
        track.candidate_confidences.append(confidence)
        if t - track.candidate_since >= self.pose_cfg.hold_seconds:
            track.posture = posture
            track.posture_confidence = round(sum(track.candidate_confidences) / len(track.candidate_confidences), 3)

    def _identity(self, track: _Track) -> tuple[str, float]:
        voted = self._vote(track)
        if voted is not None:
            track.confirmed = voted  # 새로 확정 (다른 가족으로 바뀌는 경우 포함)
        return track.confirmed or (IDENTITY_UNKNOWN, 0.0)

    def _vote(self, track: _Track) -> tuple[str, float] | None:
        """최근 얼굴 판단으로 가족 한 명이 확정되면 (이름, 신뢰도). 아니면 None."""
        if len(track.votes) < self.face_cfg.vote_min:
            return None
        name, count = Counter(v[0] for v in track.votes).most_common(1)[0]
        ratio = count / len(track.votes)
        if name == IDENTITY_UNKNOWN or ratio < self.face_cfg.vote_ratio:
            return None
        mean_conf = sum(c for n, c in track.votes if n == name) / count
        return name, round(ratio * mean_conf, 3)
