"""가짜 사람 거르기: 가족으로 확인되지 않은 사람은 CLIP에게 "사람인가, 가구·물건인가"를 물어 확인한다.

2026-11-10 실측: 자세 모델(MediaPipe)이 빈 의자를 사람으로 잡음. 관절이 보이는 정도(visibility)는 진짜 사람과 같은 0.99라
그 값으로는 거를 수 없었고, 진짜 사람과 달리 얼굴이 한 번도 잡히지 않았다.

규칙 (사람마다, track 단위):
- 가족으로 신원이 확정된 적 있음 → 사람으로 인정. 이후 같은 track 동안 다시 묻지 않음
  (얼굴이 잡힌 것만으로는 부족: 이불로 진짜 몸이 안 잡힐 때 그 얼굴이 옆의 가짜 몸에 붙은 적 있음, 2026-11-10)
- 그 밖 → interval_seconds마다 CLIP. 최근 average_count번 "사람" 확률 평균이 threshold 이상이고
  최소 min_checks번 확인했을 때만 사람 목록에 넣는다 (등 돌린 사람은 1~2초 늦게 나타남)
- CLIP을 쓸 수 없으면 거르지 않는다 (진짜 사람을 놓치는 것이 더 나쁨)
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from schema.enums import IDENTITY_UNKNOWN
from vision.activity import crop_person
from vision.clip import ClipModel
from vision.config import PersonCheckConfig
from vision.tracker import VisionPerson


@dataclass
class _State:
    known: bool = False                   # 가족으로 신원이 확정된 적 있음
    probs: deque = field(default_factory=deque)
    last_check: float = -float("inf")

    @property
    def mean(self) -> float | None:
        return sum(self.probs) / len(self.probs) if self.probs else None


class ClipPersonScorer:
    """person_probability = "사람" 문장들과 "물건" 문장들 사이 softmax에서 "사람" 쪽의 합."""

    def __init__(self, config: PersonCheckConfig, clip: ClipModel):
        self._n_person = len(config.person_prompts)
        self.prompt_set = clip.prompt_set(list(config.person_prompts) + list(config.object_prompts))
        self.clip = clip

    def __call__(self, crop_bgr: np.ndarray) -> float:
        probs = self.clip.probabilities(crop_bgr, self.prompt_set)
        return round(float(probs[: self._n_person].sum()), 3)


class PersonVerifier:
    def __init__(self, config: PersonCheckConfig, score: Callable[[np.ndarray], float] | None):
        self.config = config
        self.score = score
        self._states: dict[int, _State] = {}

    def filter(self, frame: np.ndarray, persons: list[VisionPerson], t: float) -> list[VisionPerson]:
        kept = []
        for p in persons:
            s = self._states.setdefault(p.track_id, _State())
            if p.identity != IDENTITY_UNKNOWN:
                s.known = True
            if s.known or self.score is None:
                kept.append(p)
                continue
            if t - s.last_check >= self.config.interval_seconds:
                s.last_check = t
                crop = crop_person(frame, p.bbox, self.config.crop_margin)
                if crop is not None:
                    s.probs.append(self.score(crop))
                    while len(s.probs) > self.config.average_count:
                        s.probs.popleft()
            p.person_probability = None if s.mean is None else round(s.mean, 3)
            if len(s.probs) >= self.config.min_checks and s.mean >= self.config.threshold:
                kept.append(p)
        for track_id in self._states.keys() - {p.track_id for p in persons}:
            del self._states[track_id]
        return kept
