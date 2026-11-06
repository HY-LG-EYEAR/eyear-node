"""config/vision.yaml 읽기."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VISION_CONFIG = ROOT / "config" / "vision.yaml"


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720


@dataclass
class LightConfig:
    dark_below: float = 40
    bright_above: float = 100


@dataclass
class FaceConfig:
    model: str = "buffalo_l"
    det_size: int = 640
    min_det_score: float = 0.6
    min_face_px: int = 40
    match_threshold: float = 0.40
    ambiguous_margin: float = 0.05
    vote_window: int = 10
    vote_min: int = 3
    vote_ratio: float = 0.6


@dataclass
class PoseConfig:
    model_path: str = "data/models/pose_landmarker_full.task"
    num_poses: int = 2
    min_visibility: float = 0.5
    lying_torso_deg: float = 60
    upright_torso_deg: float = 35
    standing_knee_deg: float = 150
    sitting_knee_deg: float = 120
    standing_thigh_deg: float = 30
    sitting_thigh_deg: float = 55
    walking_speed: float = 0.8
    hold_seconds: float = 0.5


@dataclass
class TrackingConfig:
    iou_match: float = 0.3
    lost_seconds: float = 1.0


@dataclass
class VisionConfig:
    camera: CameraConfig
    light: LightConfig
    face: FaceConfig
    pose: PoseConfig
    tracking: TrackingConfig

    @classmethod
    def default(cls) -> "VisionConfig":
        return cls(CameraConfig(), LightConfig(), FaceConfig(), PoseConfig(), TrackingConfig())


def _build(section_cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(section_cls)}
    unknown = values.keys() - known
    if unknown:
        raise ValueError(f"{section_cls.__name__}: 알 수 없는 설정 {sorted(unknown)}")
    return section_cls(**values)


def load_vision_config(path: Path | str = DEFAULT_VISION_CONFIG) -> VisionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = VisionConfig(
        camera=_build(CameraConfig, data.get("camera")),
        light=_build(LightConfig, data.get("light")),
        face=_build(FaceConfig, data.get("face")),
        pose=_build(PoseConfig, data.get("pose")),
        tracking=_build(TrackingConfig, data.get("tracking")),
    )
    # 상대 경로는 저장소 기준
    if not Path(config.pose.model_path).is_absolute():
        config.pose.model_path = str(ROOT / config.pose.model_path)
    return config
