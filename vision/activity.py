"""휴식(RESTING) 판단: 자세(Pose) LYING + MobileCLIP 제로샷 "쉬는 중"이 둘 다 맞고 몇 초 유지될 때만.

    python -m vision.activity --image a.jpg [b.jpg ...]   # 사진마다 문장별 확률 (문장·문턱값 조정용)

규칙 (사람마다):
    STANDING / SITTING / WALKING           → OTHER (신뢰도 = 자세 신뢰도). CLIP은 돌리지 않음
    LYING + rest p 평균 ≥ threshold, hold_seconds 유지 → RESTING (신뢰도 = min(자세 신뢰도, rest p 평균))
    LYING + rest p 평균 ≥ threshold, 아직 유지 시간 전 → 직전 값 유지
    LYING + rest p 평균 < threshold (둘이 엇갈림)  → UNKNOWN (추측하지 않음, R-13)
    rest p 평균 = 이번에 누운 동안의 최근 average_count번 CLIP 결과 평균
      (2026-11-10 실측: 한 장씩 보면 0.45~0.76으로 흔들려 기준선에서 타이머가 계속 초기화됨)
    자세 UNKNOWN (관절 안 보임, 이불 등)     → UNKNOWN
    단, RESTING 중에는 일어난 자세가 exit_seconds 이상 이어질 때만 OTHER로 바뀐다 (뒤척임으로 깜빡이지 않게).
    CLIP 모델을 쓸 수 없으면 rest p가 없으므로 LYING이어도 UNKNOWN (R-09).

판단 규칙(ActivityJudge)과 모델(ClipRestScorer, vision/clip.py)을 나눠서 규칙은 모델 없이 테스트하고, 모델은 교체할 수 있다 (R-05).
화면은 메모리에서만 쓰고 저장하지 않는다 (R-10).
"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from schema.enums import Activity, Posture
from vision.clip import ClipModel
from vision.config import RestingConfig, load_vision_config

UPRIGHT = (Posture.STANDING, Posture.SITTING, Posture.WALKING)


class RestScorer(Protocol):
    def rest_probability(self, crop_bgr: np.ndarray) -> float: ...


@dataclass
class _State:
    activity: Activity = Activity.UNKNOWN
    confidence: float = 0.0
    recent_probs: deque = field(default_factory=deque)  # 이번 "누움" 동안의 최근 CLIP 결과
    last_clip_at: float = -float("inf")
    agree_since: float | None = None      # LYING + CLIP 동의가 시작된 시각

    @property
    def mean_prob(self) -> float | None:
        return sum(self.recent_probs) / len(self.recent_probs) if self.recent_probs else None
    upright_since: float | None = None    # RESTING 중 일어난 자세가 시작된 시각


@dataclass
class ActivityResult:
    activity: Activity
    confidence: float
    rest_probability: float | None
    agree_seconds: float | None           # LYING + CLIP 동의가 이어진 시간 (화면 표시용)


class ActivityJudge:
    def __init__(self, config: RestingConfig):
        self.config = config
        self._states: dict[int, _State] = {}

    def needs_clip(self, track_id: int, posture: Posture, t: float) -> bool:
        state = self._states.get(track_id)
        last = state.last_clip_at if state else -float("inf")
        return posture == Posture.LYING and t - last >= self.config.interval_seconds

    def update(
        self, track_id: int, posture: Posture, posture_confidence: float, rest_prob: float | None, t: float
    ) -> ActivityResult:
        s = self._states.setdefault(track_id, _State())
        if rest_prob is not None:
            s.recent_probs.append(rest_prob)
            while len(s.recent_probs) > self.config.average_count:
                s.recent_probs.popleft()
            s.last_clip_at = t

        if posture in UPRIGHT:
            s.recent_probs.clear()  # 다시 누우면 새로 판단
            s.last_clip_at = -float("inf")
            s.agree_since = None
            if s.activity == Activity.RESTING:
                s.upright_since = t if s.upright_since is None else s.upright_since
                if t - s.upright_since >= self.config.exit_seconds:
                    s.activity, s.confidence, s.upright_since = Activity.OTHER, posture_confidence, None
            else:
                s.activity, s.confidence = Activity.OTHER, posture_confidence
            return self._result(s, t)

        s.upright_since = None
        mean_prob = s.mean_prob
        agrees = posture == Posture.LYING and mean_prob is not None and mean_prob >= self.config.threshold
        if agrees:
            s.agree_since = t if s.agree_since is None else s.agree_since
            if s.activity == Activity.RESTING or t - s.agree_since >= self.config.hold_seconds:
                s.activity = Activity.RESTING
                s.confidence = round(min(posture_confidence, mean_prob), 3)
            # 유지 시간 전: 직전 값 유지
        else:
            s.agree_since = None
            if s.activity != Activity.RESTING:  # RESTING은 일어난 자세로만 끝남
                s.activity, s.confidence = Activity.UNKNOWN, 0.0
        return self._result(s, t)

    def forget(self, alive_track_ids: set[int]) -> None:
        """화면에서 사라진 사람의 판단 초기화 → 다시 들어와 누우면 새 휴식."""
        for track_id in self._states.keys() - alive_track_ids:
            del self._states[track_id]

    @staticmethod
    def _result(s: _State, t: float) -> ActivityResult:
        agree = None if s.agree_since is None else t - s.agree_since
        mean = None if s.mean_prob is None else round(s.mean_prob, 3)
        return ActivityResult(s.activity, s.confidence, mean, agree)


def crop_person(frame: np.ndarray, bbox, margin: float) -> np.ndarray | None:
    """사람 상자를 margin만큼 넓혀 자름. 너무 작으면 None."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return frame[y1:y2, x1:x2]


class ClipRestScorer:
    """rest_probability = "쉬는 중" 문장들과 "아님" 문장들 사이 softmax에서 "쉬는 중" 쪽의 합."""

    def __init__(self, config: RestingConfig, clip: ClipModel):
        self.clip = clip
        self._n_rest = len(config.resting_prompts)
        self.prompt_set = clip.prompt_set(list(config.resting_prompts) + list(config.other_prompts))
        self.prompts = self.prompt_set.prompts

    def probabilities(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.clip.probabilities(crop_bgr, self.prompt_set)

    def rest_probability(self, crop_bgr: np.ndarray) -> float:
        return round(float(self.probabilities(crop_bgr)[: self._n_rest].sum()), 3)


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="사진 한 장의 CLIP 문장별 확률 (휴식 판단 문장·문턱값 조정용)")
    parser.add_argument("--image", nargs="+", required=True)
    args = parser.parse_args()

    config = load_vision_config().resting
    scorer = ClipRestScorer(config, ClipModel(config.model, config.pretrained, config.device))
    for path in args.image:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: 읽을 수 없음")
            continue
        probs = scorer.probabilities(frame)
        rest = float(probs[: len(config.resting_prompts)].sum())
        verdict = "쉬는 중에 동의" if rest >= config.threshold else "동의하지 않음"
        print(f"\n{path}: rest p = {rest:.2f} ({verdict}, 문턱값 {config.threshold})")
        for i, (prompt, p) in enumerate(zip(scorer.prompts, probs)):
            print(f"  {'휴식' if i < len(config.resting_prompts) else '아님'} {p:.2f}  {prompt}")


if __name__ == "__main__":
    main()
