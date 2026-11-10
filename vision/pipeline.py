"""카메라 프레임 한 장 → 사람들(신원·자세·활동) + 밝기. vision.demo와 main.py가 같이 쓴다.

프레임은 처리만 하고 저장하지 않는다 (R-10).
"""

from dataclasses import dataclass, field

import numpy as np

from schema.enums import Light
from vision.activity import ActivityJudge, ClipRestScorer, crop_person
from vision.clip import load_clip
from vision.config import VisionConfig, load_vision_config
from vision.continuity import SinglePersonContinuity
from vision.face_id import FaceGallery, FaceIdentifier
from vision.light import classify_light
from vision.person_check import ClipPersonScorer, PersonVerifier
from vision.pose import PoseEstimator
from vision.tracker import CameraTracker, VisionPerson


@dataclass
class VisionResult:
    persons: list[VisionPerson]
    light: Light
    lum: float | None
    hidden: list[VisionPerson] = field(default_factory=list)  # 가짜 사람으로 걸러진 것 (화면 표시·진단용, 내보내지 않음)


class VisionPipeline:
    def __init__(self, family: frozenset[str], config: VisionConfig | None = None, resting: bool = True):
        self.config = config or load_vision_config()
        if not resting:
            self.config.resting.enabled = False
        gallery = FaceGallery.load(family)
        if not gallery.embeddings:
            print("등록된 얼굴이 없음 → 모든 사람이 UNKNOWN으로 나옴. 등록: python -m vision.enroll --name MEMBER_A")
        else:
            print("등록된 얼굴:", ", ".join(f"{n}({len(e)}장)" for n, e in gallery.embeddings.items()))
        print("카메라 모델 불러오는 중...")
        self.faces = FaceIdentifier(gallery, self.config.face)
        self.pose = PoseEstimator(self.config.pose)
        self.tracker = CameraTracker(self.config.face, self.config.pose, self.config.tracking)
        clip = load_clip(self.config.resting)
        self.scorer = ClipRestScorer(self.config.resting, clip) if clip else None
        check = self.config.person_check
        self.verifier = PersonVerifier(check, ClipPersonScorer(check, clip) if clip and check.enabled else None)
        self.activity = ActivityJudge(self.config.resting)
        tracking = self.config.tracking
        self.single = SinglePersonContinuity(tracking.single_person_hold_seconds) if tracking.single_person else None

    def process(self, frame: np.ndarray, now: float) -> VisionResult:
        light, lum = classify_light(frame, self.config.light)
        faces = self.faces.identify(frame)
        poses = self.pose.detect(frame, now)
        tracked = self.tracker.update(faces, poses, now)
        persons = self.verifier.filter(frame, tracked, now)  # 얼굴 없는 가짜 사람(빈 의자 등) 거르기
        kept = {id(p) for p in persons}
        hidden = [p for p in tracked if id(p) not in kept]
        if self.single is not None:  # 1인 방: 카메라 트랙이 바뀌어도 같은 사람으로 (휴식 판단도 이어짐)
            persons = self.single.update(persons, now)
        for p in persons:
            rest_prob = None
            if self.scorer is not None and self.activity.needs_clip(p.track_id, p.posture, now):
                crop = crop_person(frame, p.bbox, self.config.resting.crop_margin)
                if crop is not None:
                    rest_prob = self.scorer.rest_probability(crop)
            judged = self.activity.update(p.track_id, p.posture, p.posture_confidence, rest_prob, now)
            p.activity, p.activity_confidence = judged.activity, judged.confidence
            p.rest_probability, p.rest_agree_seconds = judged.rest_probability, judged.agree_seconds
        self.activity.forget({p.track_id for p in persons})
        return VisionResult(persons, light, lum, hidden)

    def close(self) -> None:
        self.pose.close()
