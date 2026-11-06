"""자세 추정: MediaPipe Pose 관절 좌표(사전 학습 모델) → 규칙으로 자세 판단.

한 프레임 판단(classify_posture)과 모델 실행(PoseEstimator)을 나눠서, 규칙은 모델 없이 테스트할 수 있다.
WALKING은 여러 프레임의 이동 속도가 필요하므로 vision/tracker.py에서 판단한다.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from schema.enums import Posture
from vision.config import PoseConfig

MODEL_URL_HINT = (
    "mkdir -p data/models && curl -fL -o data/models/pose_landmarker_full.task "
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# MediaPipe Pose 관절 번호 (mediapipe.tasks.python.vision.PoseLandmark)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


@dataclass
class PoseResult:
    points: np.ndarray       # (33, 2) 픽셀 좌표 x, y (y는 아래로 증가)
    visibility: np.ndarray   # (33,) 0~1
    bbox: tuple[float, float, float, float]  # 보이는 관절을 감싸는 상자

    def visible(self, index: int, min_visibility: float) -> bool:
        return bool(self.visibility[index] >= min_visibility)


@dataclass
class PostureJudgement:
    posture: Posture
    confidence: float
    torso_center: tuple[float, float] | None  # 이동 속도 계산용 (엉덩이 중심)
    torso_length: float | None                # 화면 속 몸 크기 (속도 정규화용)


def _angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    dx, dy = top - bottom
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _joint_angle(a: np.ndarray, joint: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - joint, b - joint
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _mid(pose: PoseResult, left: int, right: int, min_vis: float) -> tuple[np.ndarray | None, list[int]]:
    """좌우 중 보이는 관절의 중점. 둘 다 안 보이면 None."""
    used = [i for i in (left, right) if pose.visible(i, min_vis)]
    if not used:
        return None, []
    return pose.points[used].mean(axis=0), used


def classify_posture(pose: PoseResult, config: PoseConfig) -> PostureJudgement:
    """한 프레임의 자세. 필요한 관절이 안 보이면 UNKNOWN (추측하지 않음)."""
    v = config.min_visibility
    shoulder, s_used = _mid(pose, L_SHOULDER, R_SHOULDER, v)
    hip, h_used = _mid(pose, L_HIP, R_HIP, v)
    if shoulder is None or hip is None:
        return PostureJudgement(Posture.UNKNOWN, 0.0, None, None)

    torso_len = float(np.linalg.norm(shoulder - hip))
    center = (float(hip[0]), float(hip[1]))
    torso_deg = _angle_from_vertical(shoulder, hip)
    used = s_used + h_used

    def judged(posture: Posture, extra: list[int]) -> PostureJudgement:
        confidence = float(np.mean(pose.visibility[used + extra]))
        return PostureJudgement(posture, round(min(0.95, confidence), 3), center, torso_len)

    if torso_deg >= config.lying_torso_deg:
        return judged(Posture.LYING, [])
    if torso_deg > config.upright_torso_deg:
        return PostureJudgement(Posture.UNKNOWN, 0.0, center, torso_len)  # 눕는 중/일어나는 중

    # 몸통이 서 있음 → 다리로 서기/앉기 구분. 무릎 각도(발목 필요)를 우선, 없으면 허벅지 기울기
    knee_angles, thigh_angles, leg_used = [], [], []
    for hip_i, knee_i, ankle_i in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        if not (pose.visible(hip_i, v) and pose.visible(knee_i, v)):
            continue
        p = pose.points
        thigh_angles.append(_angle_from_vertical(p[hip_i], p[knee_i]))
        leg_used.append(knee_i)
        if pose.visible(ankle_i, v):
            knee_angles.append(_joint_angle(p[hip_i], p[knee_i], p[ankle_i]))
            leg_used.append(ankle_i)

    if knee_angles:
        knee = float(np.mean(knee_angles))
        if knee >= config.standing_knee_deg:
            return judged(Posture.STANDING, leg_used)
        if knee <= config.sitting_knee_deg:
            return judged(Posture.SITTING, leg_used)
    elif thigh_angles:
        thigh = float(np.mean(thigh_angles))
        if thigh <= config.standing_thigh_deg:
            return judged(Posture.STANDING, leg_used)
        if thigh >= config.sitting_thigh_deg:
            return judged(Posture.SITTING, leg_used)
    # 다리가 안 보이거나 애매함 (예: 책상 앞 노트북 카메라)
    return PostureJudgement(Posture.UNKNOWN, 0.0, center, torso_len)


class PoseEstimator:
    def __init__(self, config: PoseConfig):
        import mediapipe as mp  # 무거워서 실제로 쓸 때만 불러옴
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(config.model_path).exists():
            raise FileNotFoundError(f"자세 모델 파일 없음: {config.model_path}\n받는 방법: {MODEL_URL_HINT}")

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=config.num_poses,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._config = config
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> list[PoseResult]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(timestamp_s * 1000), self._last_ms + 1)  # VIDEO 모드는 시각이 계속 증가해야 함
        self._last_ms = ms
        result = self._landmarker.detect_for_video(image, ms)

        poses = []
        for landmarks in result.pose_landmarks:
            points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            visibility = np.array([lm.visibility or 0.0 for lm in landmarks], dtype=np.float32)
            seen = points[visibility >= self._config.min_visibility]
            if len(seen) == 0:
                continue
            x1, y1 = seen.min(axis=0)
            x2, y2 = seen.max(axis=0)
            poses.append(PoseResult(points, visibility, (float(x1), float(y1), float(x2), float(y2))))
        return poses

    def close(self) -> None:
        self._landmarker.close()
