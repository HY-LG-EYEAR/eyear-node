"""카메라 프레임 한 장 → 사람들(신원·자세) + 밝기. vision.demo와 main.py가 같이 쓴다.

프레임은 처리만 하고 저장하지 않는다 (R-10).
"""

from dataclasses import dataclass

import numpy as np

from schema.enums import Light
from vision.config import VisionConfig, load_vision_config
from vision.face_id import FaceGallery, FaceIdentifier
from vision.light import classify_light
from vision.pose import PoseEstimator
from vision.tracker import CameraTracker, VisionPerson


@dataclass
class VisionResult:
    persons: list[VisionPerson]
    light: Light
    lum: float | None


class VisionPipeline:
    def __init__(self, family: frozenset[str], config: VisionConfig | None = None):
        self.config = config or load_vision_config()
        gallery = FaceGallery.load(family)
        if not gallery.embeddings:
            print("등록된 얼굴이 없음 → 모든 사람이 UNKNOWN으로 나옴. 등록: python -m vision.enroll --name MEMBER_A")
        else:
            print("등록된 얼굴:", ", ".join(f"{n}({len(e)}장)" for n, e in gallery.embeddings.items()))
        print("카메라 모델 불러오는 중...")
        self.faces = FaceIdentifier(gallery, self.config.face)
        self.pose = PoseEstimator(self.config.pose)
        self.tracker = CameraTracker(self.config.face, self.config.pose, self.config.tracking)

    def process(self, frame: np.ndarray, now: float) -> VisionResult:
        light, lum = classify_light(frame, self.config.light)
        faces = self.faces.identify(frame)
        poses = self.pose.detect(frame, now)
        persons = self.tracker.update(faces, poses, now)
        return VisionResult(persons, light, lum)

    def close(self) -> None:
        self.pose.close()
