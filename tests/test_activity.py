"""휴식(RESTING) 판단 규칙 테스트 (모델 없이, 시각을 직접 넘김)."""

import numpy as np
import pytest

from fusion.room_state import build_snapshot
from schema.enums import Activity as A
from schema.enums import Light
from schema.enums import Posture as P
from schema.family import load_family
from vision.activity import ActivityJudge, crop_person
from vision.clip import pad_square
from vision.config import RestingConfig, load_vision_config
from vision.tracker import VisionPerson

CFG = RestingConfig()  # threshold 0.6, interval 1초, hold 5초, exit 2초
STEP = 0.1


class Run:
    """0.1초 간격 프레임을 흉내 낸다. CLIP은 needs_clip일 때만 rest_prob을 낸다."""

    def __init__(self):
        self.judge = ActivityJudge(CFG)
        self.i = 0  # 프레임 번호 (0.1초를 계속 더하면 소수점 오차가 쌓여서 번호로 시각 계산)
        self.clip_calls = 0
        self.last = None

    def frames(self, posture, seconds, rest_prob=0.8, track_id=1, confidence=0.9):
        for _ in range(round(seconds / STEP)):
            t = self.i * STEP
            prob = None
            if self.judge.needs_clip(track_id, posture, t):
                prob = rest_prob
                self.clip_calls += 1
            self.last = self.judge.update(track_id, posture, confidence, prob, t)
            self.i += 1
        return self.last.activity


def test_upright_is_other_without_clip():
    r = Run()
    assert r.frames(P.STANDING, 1) == A.OTHER and r.last.confidence == 0.9
    assert r.frames(P.SITTING, 1) == A.OTHER
    assert r.clip_calls == 0


def test_lying_and_clip_agree_for_hold_seconds_becomes_resting():
    r = Run()
    r.frames(P.STANDING, 1)
    assert r.frames(P.LYING, 4.8) == A.OTHER          # 아직 5초 전 → 직전 값 유지
    assert r.frames(P.LYING, 0.4) == A.RESTING
    assert r.last.confidence == pytest.approx(0.8)    # min(자세 0.9, CLIP 평균 0.8)
    assert r.clip_calls == 6                          # 누운 동안 1초에 한 번만


def test_lying_but_clip_disagrees_is_unknown():
    r = Run()
    r.frames(P.STANDING, 1)
    assert r.frames(P.LYING, 10, rest_prob=0.3) == A.UNKNOWN


def test_unknown_posture_resets_hold_timer():
    r = Run()
    r.frames(P.LYING, 3)
    r.frames(P.UNKNOWN, 0.5)
    assert r.frames(P.LYING, 3) == A.UNKNOWN          # 3초 + 3초지만 중간에 끊김
    assert r.frames(P.LYING, 2.2) == A.RESTING


def test_resting_survives_tossing_and_clip_drop():
    r = Run()
    r.frames(P.LYING, 5.2)
    assert r.last.activity == A.RESTING
    assert r.frames(P.UNKNOWN, 3) == A.RESTING        # 이불로 관절이 안 보임
    assert r.frames(P.SITTING, 1.5) == A.RESTING      # 잠깐 뒤척임 (2초 미만)
    assert r.frames(P.LYING, 3, rest_prob=0.3) == A.RESTING  # CLIP이 흔들려도 유지


def test_getting_up_for_exit_seconds_ends_resting():
    r = Run()
    r.frames(P.LYING, 5.2)
    assert r.frames(P.SITTING, 1.8) == A.RESTING
    assert r.frames(P.STANDING, 0.3) == A.OTHER
    # 다시 누우면 새 휴식 (CLIP도 새로 판단, 5초 다시 기다림)
    calls = r.clip_calls
    assert r.frames(P.LYING, 4.8) == A.OTHER
    assert r.frames(P.LYING, 0.4) == A.RESTING
    assert r.clip_calls > calls


def test_forget_resets_person_who_left():
    r = Run()
    r.frames(P.LYING, 5.2)
    r.judge.forget(set())
    assert r.frames(P.LYING, 1) == A.UNKNOWN          # 처음부터 다시: 5초 기다림


def test_no_model_means_lying_is_unknown():
    judge = ActivityJudge(CFG)
    for i in range(100):  # 10초 누워 있어도 CLIP 결과가 없으면 RESTING을 지어내지 않음
        result = judge.update(1, P.LYING, 0.9, None, i * STEP)
    assert result.activity == A.UNKNOWN


def test_crop_and_pad():
    frame = np.zeros((100, 200, 3), np.uint8)
    crop = crop_person(frame, (50, 40, 150, 60), margin=0.3)
    assert crop.shape[:2] == (32, 160)
    assert crop_person(frame, (0, 0, 10, 10), margin=0.3) is None
    assert pad_square(crop).shape == (160, 160, 3)


def test_room_state_sends_activity():
    person = VisionPerson(track_id=1, bbox=(0, 0, 10, 10), identity="MEMBER_A", identity_confidence=0.9,
                          posture=P.LYING, posture_confidence=0.85, activity=A.RESTING, activity_confidence=0.8)
    snap = build_snapshot([person], Light.BRIGHT, None, load_family())
    assert (snap.persons[0].activity, snap.persons[0].activity_confidence) == ("RESTING", 0.8)


def test_config_file_resting_section():
    config = load_vision_config().resting
    assert config.resting_prompts and config.other_prompts and 0 < config.threshold < 1


@pytest.mark.slow
def test_real_mobileclip_outputs_probability():
    """실제 MobileCLIP을 불러와 확률이 0~1인지 확인 (정확도는 사람이 직접 측정). pytest -m slow 로 실행."""
    from vision.activity import ClipRestScorer
    from vision.clip import ClipModel

    config = load_vision_config().resting
    scorer = ClipRestScorer(config, ClipModel(config.model, config.pretrained, config.device))
    image = np.random.default_rng(0).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    probs = scorer.probabilities(image)
    assert len(probs) == len(config.resting_prompts) + len(config.other_prompts)
    assert probs.sum() == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= scorer.rest_probability(image) <= 1.0


def test_fluctuating_clip_uses_average():
    """2026-11-10 실측처럼 한 장씩은 0.45~0.65로 흔들려도 평균이 기준(0.5) 이상이면 RESTING."""
    judge = ActivityJudge(CFG)
    probs = [0.45, 0.62, 0.55, 0.48, 0.64, 0.47, 0.58, 0.52]
    result, k = None, 0
    for i in range(80):  # 0.1초 간격 8초
        t = i * STEP
        prob = None
        if judge.needs_clip(1, P.LYING, t):
            prob, k = probs[k % len(probs)], k + 1
        result = judge.update(1, P.LYING, 0.9, prob, t)
    assert result.activity == A.RESTING


def test_person_verifier_filters_faceless_objects():
    from vision.config import PersonCheckConfig
    from vision.face_id import FaceMatch
    from vision.person_check import PersonVerifier

    frame = np.zeros((400, 400, 3), np.uint8)

    def person(track_id, face=False, identity="UNKNOWN"):
        f = FaceMatch((10, 10, 60, 70), identity, 0.6, 0.8) if face else None
        return VisionPerson(track_id=track_id, bbox=(50, 50, 250, 250), identity=identity, identity_confidence=0.0,
                            posture=P.UNKNOWN, posture_confidence=0.0, face=f)

    config = PersonCheckConfig()  # 1초마다, 최소 2번, 평균 0.5
    chair = PersonVerifier(config, lambda crop: 0.2)
    real = PersonVerifier(config, lambda crop: 0.8)
    for t in (0.0, 1.0, 2.0, 3.0):
        assert chair.filter(frame, [person(1)], t) == []          # 빈 의자: 끝까지 안 나타남
    assert real.filter(frame, [person(1)], 0.0) == []              # 등 돌린 사람: 첫 확인만으로는 아직
    assert len(real.filter(frame, [person(1)], 1.0)) == 1          # 두 번째 확인에서 나타남
    assert chair.filter(frame, [person(3, face=True)], 0.0) == []    # 모르는 얼굴만으로는 인정하지 않음
    assert len(chair.filter(frame, [person(2, face=True, identity="MEMBER_C")], 0.0)) == 1  # 가족이면 묻지 않음
    assert len(chair.filter(frame, [person(2)], 1.0)) == 1         # 같은 track은 이후 신원이 안 보여도 유지
    assert len(PersonVerifier(config, None).filter(frame, [person(1)], 0.0)) == 1  # CLIP 없으면 거르지 않음
