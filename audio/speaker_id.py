"""목소리로 누구인지 알아보기 (화자 식별). 사전 학습 모델만 씀, 직접 학습 없음.

- 모델: 3D-Speaker CAM++ (중국어·영어 학습, sherpa-onnx로 실행, 문장 하나 약 3ms, torch 불필요)
  data/models/speaker/ 에 내려받음 (README의 설치 안내). 같은 인터페이스면 다른 모델로 바꿀 수 있음 (R-05)
- 등록: python -m audio.enroll_voice --name MEMBER_D  → data/voices/<이름>.npy 에 목소리 특징값만 저장.
  녹음은 파일로 남기지 않고 특징값을 뽑은 뒤 버림 (R-10, 얼굴 등록과 같은 방식)
- 판단: 등록된 사람마다 특징값 평균과의 코사인 유사도. 가장 높은 값이 threshold 이상이고 두 번째와 margin 이상 차이 날 때만
  그 사람. 아니면 UNKNOWN (지어내지 않음, R-13). min_seconds보다 짧은 말("응")은 판단하지 않음
- 2026-11-23 동작 확인 (macOS 합성 음성 8개): 1.9초 명령은 같은 목소리 0.79~0.89 / 다른 목소리 최대 0.58 → 8/8,
  0.5초 "Yes"는 같은 0.44~0.69 / 다른 최대 0.44로 겹침. 사람 목소리 기준값은 팀원 녹음으로 다시 정할 것
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from schema.enums import IDENTITY_UNKNOWN

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "data" / "voices"
SAMPLE_RATE = 16000


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def to_float(audio: np.ndarray) -> np.ndarray:
    """int16 → float32 (-1~1). 이미 float면 그대로."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


class OnnxSpeakerEmbedder:
    """sherpa-onnx 화자 특징값 추출기. audio(16kHz 모노) → 정규화된 특징값."""

    def __init__(self, model_path: Path | str, num_threads: int = 2):
        import sherpa_onnx  # 실제로 쓸 때만 불러옴 (팀원은 설치 불필요)

        path = Path(model_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"화자 인식 모델 없음: {path} (README의 화자 식별 설치 안내 참고)")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=num_threads))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, to_float(audio))
        stream.input_finished()
        return _normalize(np.asarray(self._extractor.compute(stream), dtype=np.float32))


class VoiceGallery:
    """등록된 목소리 특징값 (사람마다 여러 개). 파일: <dir>/<이름>.npy, 모양 (개수, 차원)."""

    def __init__(self, directory: Path | str = VOICE_DIR):
        self.directory = Path(directory)
        self.samples: dict[str, np.ndarray] = {}
        if self.directory.exists():
            for f in sorted(self.directory.glob("*.npy")):
                self.samples[f.stem] = np.load(f)

    def save(self, name: str, embeddings: np.ndarray, append: bool = False) -> int:
        if append and name in self.samples:
            embeddings = np.concatenate([self.samples[name], embeddings])
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / f"{name}.npy", embeddings)
        self.samples[name] = embeddings
        return len(embeddings)

    def delete(self, name: str) -> bool:
        f = self.directory / f"{name}.npy"
        self.samples.pop(name, None)
        if f.exists():
            f.unlink()
            return True
        return False

    def centroids(self) -> dict[str, np.ndarray]:
        return {name: _normalize(np.mean([_normalize(e) for e in embs], axis=0)) for name, embs in self.samples.items()}


class SpeakerIdentifier:
    def __init__(self, embed: Callable[[np.ndarray], np.ndarray], gallery: VoiceGallery,
                 threshold: float = 0.65, margin: float = 0.05, min_seconds: float = 1.0):
        self.embed = embed
        self.centroids = gallery.centroids()
        self.threshold = threshold
        self.margin = margin
        self.min_seconds = min_seconds

    def scores(self, audio: np.ndarray) -> dict[str, float]:
        """사람마다 코사인 유사도 (등록 확인·측정용)."""
        e = self.embed(audio)
        return {name: float(e @ c) for name, c in self.centroids.items()}

    def identify(self, audio: np.ndarray) -> tuple[str, float]:
        """(이름, 신뢰도). 판단할 수 없으면 (UNKNOWN, 0.0)."""
        if not self.centroids or len(audio) < self.min_seconds * SAMPLE_RATE:
            return IDENTITY_UNKNOWN, 0.0
        ranked = sorted(self.scores(audio).items(), key=lambda kv: kv[1], reverse=True)
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if best < self.threshold or best - second < self.margin:
            return IDENTITY_UNKNOWN, 0.0
        return best_name, round(min(best, 1.0), 2)


def load_identifier(config) -> SpeakerIdentifier | None:
    """설정대로 화자 식별기를 만든다. 꺼져 있거나 모델·등록이 없으면 None (안내만 출력)."""
    if not config.enabled:
        return None
    gallery = VoiceGallery(ROOT / config.voices_dir)
    if not gallery.samples:
        print("화자 식별: 등록된 목소리 없음 → 끔 (python -m audio.enroll_voice --name MEMBER_X)")
        return None
    try:
        embed = OnnxSpeakerEmbedder(config.model_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"화자 식별: 끔 ({e})")
        return None
    print(f"화자 식별 켜짐: {', '.join(sorted(gallery.samples))}")
    return SpeakerIdentifier(embed, gallery, config.threshold, config.margin, config.min_seconds)
