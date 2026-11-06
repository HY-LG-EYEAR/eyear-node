"""말소리 구간 자르기 (webrtcvad). 말소리 구간만 음성 인식에 넘겨 Whisper의 무음 환각을 막는다."""

from collections import deque

import numpy as np

from audio.config import VadConfig
from audio.mic import FRAME_SAMPLES, SAMPLE_RATE

FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


class SpeechSegmenter:
    """프레임을 하나씩 push하면, 말 하나가 끝났을 때 그 구간의 소리(int16)를 돌려준다."""

    def __init__(self, config: VadConfig, is_speech=None):
        if is_speech is None:
            import webrtcvad

            vad = webrtcvad.Vad(config.aggressiveness)
            is_speech = lambda frame: vad.is_speech(frame.tobytes(), SAMPLE_RATE)  # noqa: E731
        self._is_speech = is_speech
        self.config = config
        self._pre_roll: deque[np.ndarray] = deque(maxlen=max(1, config.pre_roll_ms // FRAME_MS))
        self._end_frames = max(1, config.end_silence_ms // FRAME_MS)
        self._max_frames = int(config.max_segment_seconds * 1000 // FRAME_MS)
        self.reset()

    def reset(self) -> None:
        self._pre_roll.clear()
        self._run = 0             # 연속된 말소리 프레임 수 (시작 판단용)
        self._segment: list[np.ndarray] = []
        self._voiced = 0          # 구간 안 말소리 프레임 수
        self._silence = 0         # 구간 안 연속 무음 프레임 수
        self.in_speech = False

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        speech = self._is_speech(frame)
        if not self.in_speech:
            self._pre_roll.append(frame)
            self._run = self._run + 1 if speech else 0
            if self._run >= self.config.start_frames:
                self.in_speech = True
                self._segment = list(self._pre_roll)
                self._voiced, self._silence = self._run, 0
            return None

        self._segment.append(frame)
        if speech:
            self._voiced += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self._end_frames or len(self._segment) >= self._max_frames:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        segment, voiced = self._segment, self._voiced
        self.reset()
        if voiced * FRAME_MS < self.config.min_speech_ms:
            return None
        return np.concatenate(segment)
