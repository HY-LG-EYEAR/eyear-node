"""음성 인식 (faster-whisper, 로컬 실행). 소리는 Node 밖으로 나가지 않고 텍스트만 나간다 (R-10)."""

import math
import re
from dataclasses import dataclass

import numpy as np

from audio.config import SttConfig

_NOT_WORD = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """비교용: 공백·문장부호를 없애고 소문자로."""
    return _NOT_WORD.sub("", text).lower()


@dataclass
class Transcript:
    text: str
    confidence: float  # 0~1


def segment_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    """Whisper 구간 신뢰도 = 토큰 평균 확률 × 말소리일 확률. 단계 8에서 실측과 비교해 보정."""
    return max(0.0, min(1.0, math.exp(avg_logprob) * (1.0 - no_speech_prob)))


def is_hallucination(text: str, patterns: list[str]) -> bool:
    norm = normalize(text)
    return any(normalize(p) in norm for p in patterns)


class SpeechRecognizer:
    def __init__(self, config: SttConfig):
        from faster_whisper import WhisperModel  # 무거워서 실제로 쓸 때만 불러옴

        self.config = config
        self._model = WhisperModel(config.model, device="cpu", compute_type=config.compute_type)

    def transcribe(self, audio_int16: np.ndarray) -> Transcript | None:
        """말소리가 아니거나 환각으로 보이면 None (지어내지 않음)."""
        # 앞뒤에 무음을 붙임: 말이 바로 시작하는 짧은 구간("응")은 Whisper가 놓치는 경우가 있음 (2026-11-06 실험)
        pad = np.zeros(int(self.config.pad_seconds * 16000), dtype=np.int16)
        audio = np.concatenate([pad, audio_int16, pad]).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language="ko",
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        segments = [s for s in segments if s.text.strip()]
        if not segments:
            return None
        text = " ".join(s.text.strip() for s in segments).strip()
        if not normalize(text) or is_hallucination(text, self.config.hallucinations):
            return None
        confidence = sum(segment_confidence(s.avg_logprob, s.no_speech_prob) for s in segments) / len(segments)
        return Transcript(text, round(confidence, 3))
