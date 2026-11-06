"""카메라 화면 속 사람 추적: 얼굴·자세 결과를 사람 단위로 묶고, 여러 프레임에 걸쳐 안정화한다.

- 신원: 최근 얼굴 판단들의 투표. 얼굴이 안 보이는 프레임은 투표하지 않으므로, 고개를 돌려도 신원이 유지된다.
- 자세: 새 자세가 hold_seconds 이상 유지돼야 바꾼다 (깜빡임 방지).
- WALKING: 서 있으면서 몸 중심이 빠르게 옆으로 이동.

여기서의 track_id는 카메라 기준 번호다. 레이더와 합치는 것은 fusion(단계 5)에서 한다.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

from schema.enums import IDENTITY_UNKNOWN, Posture
from vision.config import FaceConfig, PoseConfig, TrackingConfig
from vision.face_id import FaceMatch
from vision.pose import PoseResult, classify_posture, NOSE

BBox = tuple[float, float, float, float]


@dataclass
class VisionPerson:
    track_id: int
    bbox: BBox
    identity: str
    identity_confidence: float
    posture: Posture
    posture_confidence: float
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


def _inside(point: tuple[float, float], b: BBox) -> bool:
    return b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]


@dataclass
class _Detection:
    bbox: BBox
    face: FaceMatch | None
    pose: PoseResult | None


def pair_faces_with_poses(faces: list[FaceMatch], poses: list[PoseResult]) -> list[_Detection]:
    """얼굴을 그 얼굴이 속한 몸(코 위치가 얼굴 상자 안)에 붙인다. 몸이 없는 얼굴은 얼굴만으로 사람 하나."""
    detections = [_Detection(p.bbox, None, p) for p in poses]
    for face in faces:
        owner = next(
            (d for d in detections if d.face is None and _inside(tuple(d.pose.points[NOSE]), face.bbox)),
            None,
        )
        if owner is None:  # 코가 안 보이는 경우: 얼굴 중심이 몸 상자 안에 있으면
            owner = next((d for d in detections if d.face is None and _inside(_center(face.bbox), d.bbox)), None)
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
        detections = pair_faces_with_poses(faces, poses)
        assigned = self._assign(detections, t)
        self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self.track_cfg.lost_seconds]

        persons = []
        for track in sorted(self._tracks, key=lambda tr: tr.id):
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
        for di, d in enumerate(detections):
            if di not in used_dets:
                track = _Track(self._next_id, d.bbox, t)
                self._next_id += 1
                self._tracks.append(track)
                assigned[track.id] = d
        for track in self._tracks:
            if track.id in assigned:
                self._observe(track, assigned[track.id], t)
        return assigned

    def _observe(self, track: _Track, d: _Detection, t: float) -> None:
        track.bbox = d.bbox
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
        if len(track.votes) < self.face_cfg.vote_min:
            return IDENTITY_UNKNOWN, 0.0
        name, count = Counter(v[0] for v in track.votes).most_common(1)[0]
        ratio = count / len(track.votes)
        if name == IDENTITY_UNKNOWN or ratio < self.face_cfg.vote_ratio:
            return IDENTITY_UNKNOWN, 0.0
        mean_conf = sum(c for n, c in track.votes if n == name) / count
        return name, round(ratio * mean_conf, 3)
