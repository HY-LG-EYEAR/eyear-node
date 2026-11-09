"""소리 환경 분류: 마이크 소리 → audio.context (SILENT / SPEECH / MUSIC / ...).

    python -m audio.classifier            # 마이크로 1초마다 분류 결과 찍기 (기준값 조정용)
    python -m audio.classifier --wav f    # WAV 파일

1초 창마다:
  1. 소리 크기(dBFS)가 silent_below_dbfs보다 작으면 SILENT
  2. YAMNet(사전 학습, 521종) 점수를 config의 라벨 묶음으로 모아 가장 높은 값
     - 말·대화·음악·TV 점수가 min_score 이상이면 그 값
     - 아니고 소리가 loud_above_dbfs보다 크면 LOUD_NOISE
     - 아니고 Silence 점수가 min_score 이상이면 SILENT
     - 그 밖(키보드 소리 등 분류할 수 없는 소리)은 UNKNOWN (지어내지 않음, R-13)
  3. 최근 vote_windows개 창의 다수결 (화면 한 장 오판과 같은 1초짜리 흔들림 방지)

- Node가 말하는 동안은 마이크가 꺼져 있어 직전 값을 유지한다 (스피커 소리를 분류하지 않음).
- 마이크에서 소리가 오지 않으면 결과를 내지 않는다 → 방 상태에서 UNKNOWN이 된다 (R-09).
- 소리는 1초 창을 분류한 뒤 바로 버린다. 저장·전송하지 않는다 (R-10).
"""

import argparse
import math
import queue
import threading
from collections import Counter, deque
from collections.abc import Callable

import numpy as np

from audio.config import ClassifierConfig, load_audio_config
from audio.mic import FRAME_SECONDS, SAMPLE_RATE, _FanOut
from schema.enums import AudioContext
from schema.models import Audio

MODEL_URL_HINT = (
    "mkdir -p data/models && curl -fL -o data/models/yamnet.tflite "
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
)

# 라벨 점수로 정하는 "의미 있는 소리" (LOUD_NOISE보다 우선)
_MEANINGFUL = (AudioContext.SPEECH, AudioContext.CONVERSATION, AudioContext.MUSIC, AudioContext.TV)


def dbfs(samples: np.ndarray) -> float:
    """int16 소리의 크기 (0 = 최대, 작을수록 조용함)."""
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) / 32768.0
    return 20 * math.log10(max(rms, 1e-9))


def classify_window(scores: dict[str, float], level_dbfs: float, config: ClassifierConfig) -> tuple[AudioContext, float]:
    """창 하나의 (값, 신뢰도)."""
    if level_dbfs < config.silent_below_dbfs:
        return AudioContext.SILENT, config.level_confidence

    grouped = {
        AudioContext(ctx): max((scores.get(label, 0.0) for label in labels), default=0.0)
        for ctx, labels in config.labels.items()
    }
    meaningful = {ctx: s for ctx, s in grouped.items() if ctx in _MEANINGFUL and s >= config.min_score}
    if meaningful:
        best = max(meaningful, key=meaningful.get)
        return best, meaningful[best]
    if level_dbfs > config.loud_above_dbfs:
        return AudioContext.LOUD_NOISE, config.level_confidence
    silence = grouped.get(AudioContext.SILENT, 0.0)
    if silence >= config.min_score:
        return AudioContext.SILENT, silence
    return AudioContext.UNKNOWN, 0.0


class ContextVoter:
    """최근 n개 창의 다수결. 동점이면 더 최근 값. 신뢰도는 이긴 값들의 평균."""

    def __init__(self, n: int):
        self._window: deque[tuple[AudioContext, float]] = deque(maxlen=n)

    def push(self, context: AudioContext, confidence: float) -> Audio:
        self._window.append((context, confidence))
        counts = Counter(c for c, _ in self._window)
        top = max(counts.values())
        winner = next(c for c, _ in reversed(self._window) if counts[c] == top)
        confs = [conf for c, conf in self._window if c == winner]
        return Audio(context=winner, confidence=round(sum(confs) / len(confs), 2))

    def reset(self) -> None:
        self._window.clear()


class YamnetModel:
    """mediapipe AudioClassifier (YAMNet). scores(): 라벨 이름 → 점수 (0~1)."""

    def __init__(self, config: ClassifierConfig):
        from mediapipe.tasks.python import BaseOptions, audio  # 무거워서 실제로 쓸 때만 불러옴
        from mediapipe.tasks.python.components.containers import audio_data

        if not config.model_file.exists():
            raise FileNotFoundError(f"소리 분류 모델 파일 없음: {config.model_file}\n받는 방법: {MODEL_URL_HINT}")
        self._audio_data = audio_data
        options = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_file)),
            running_mode=audio.RunningMode.AUDIO_CLIPS,
            max_results=-1,
        )
        self._classifier = audio.AudioClassifier.create_from_options(options)
        self.labels = set(self.scores(np.zeros(SAMPLE_RATE, np.int16)))
        wrong = sorted({label for labels in config.labels.values() for label in labels} - self.labels)
        if wrong:
            raise ValueError(f"config/audio.yaml classifier.labels에 YAMNet에 없는 라벨: {wrong}")

    def scores(self, samples: np.ndarray) -> dict[str, float]:
        clip = self._audio_data.AudioData.create_from_array(samples.astype(np.float32) / 32768.0, SAMPLE_RATE)
        results = self._classifier.classify(clip)  # 0.975초 창, 0.48초 간격 → 1초에 결과 2개
        total: dict[str, float] = {}
        for result in results:
            for cat in result.classifications[0].categories:
                total[cat.category_name] = total.get(cat.category_name, 0.0) + cat.score
        return {name: s / len(results) for name, s in total.items()}

    def close(self) -> None:
        self._classifier.close()


class AudioContextWorker:
    """마이크 프레임을 1초씩 모아 분류하고, 다수결 결과를 on_result로 넘긴다 (스레드에서 run)."""

    def __init__(
        self,
        source: _FanOut,
        scores: Callable[[np.ndarray], dict[str, float]],
        config: ClassifierConfig,
        on_result: Callable[[Audio], None],
        on_window: Callable[[AudioContext, float, float, dict[str, float]], None] | None = None,
    ):
        self.source = source
        self.frames = source.subscribe()
        self.scores = scores
        self.config = config
        self.on_result = on_result
        self.on_window = on_window
        self.voter = ContextVoter(config.vote_windows)
        self._buffer: list[np.ndarray] = []
        self._frames_per_window = round(config.window_seconds / FRAME_SECONDS)
        self._latest: Audio | None = None

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.step(timeout=0.1)

    def step(self, timeout: float = 0.1) -> None:
        try:
            frame = self.frames.get(timeout=timeout) if timeout > 0 else self.frames.get_nowait()
        except queue.Empty:
            if self.source.muted:
                # Node가 말하는 중: 스피커 소리는 분류하지 않고 직전 값을 유지
                self._buffer.clear()
                if self._latest is not None:
                    self.on_result(self._latest)
            return
        self._buffer.append(frame)
        if len(self._buffer) < self._frames_per_window:
            return
        samples = np.concatenate(self._buffer)
        self._buffer.clear()
        level = dbfs(samples)
        scores = self.scores(samples)
        del samples  # 분류가 끝난 소리는 바로 버림
        context, confidence = classify_window(scores, level, self.config)
        if self.on_window:
            self.on_window(context, confidence, level, scores)
        self._latest = self.voter.push(context, confidence)
        self.on_result(self._latest)


def main() -> None:
    from audio.mic import MicStream, WavSource

    parser = argparse.ArgumentParser(description="소리 환경 분류 데모 (기준값 조정용)")
    parser.add_argument("--wav", default=None, help="마이크 대신 WAV 파일 (16kHz 모노 16비트)")
    args = parser.parse_args()

    config = load_audio_config().classifier
    model = YamnetModel(config)
    source = WavSource(args.wav) if args.wav else MicStream(load_audio_config().mic.device)

    def on_window(context, confidence, level, scores):
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        top_text = ", ".join(f"{name} {s:.2f}" for name, s in top)
        print(f"{level:6.1f} dBFS  창: {context.value:<12} {confidence:.2f}  | YAMNet: {top_text}")

    def on_result(audio: Audio):
        print(f"{'':12}→ 보내는 값: {audio.context} {audio.confidence:.2f}")

    worker = AudioContextWorker(source, model.scores, config, on_result, on_window)
    stop = threading.Event()
    source.start()
    print(f"1초마다 분류 (SILENT < {config.silent_below_dbfs} dBFS, LOUD_NOISE > {config.loud_above_dbfs} dBFS). Ctrl+C로 종료")
    try:
        while not stop.is_set():
            worker.step(timeout=0.1)
            if isinstance(source, WavSource) and source.finished.is_set() and worker.frames.empty():
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        model.close()


if __name__ == "__main__":
    main()
