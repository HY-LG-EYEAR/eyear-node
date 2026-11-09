"""config/audio.yaml 읽기."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_CONFIG = ROOT / "config" / "audio.yaml"


@dataclass
class MicConfig:
    device: int | str | None = None


@dataclass
class TtsConfig:
    voice: str = "Yuna"
    mute_tail_seconds: float = 0.3


@dataclass
class VadConfig:
    aggressiveness: int = 2
    start_frames: int = 3
    pre_roll_ms: int = 300
    end_silence_ms: int = 800
    min_speech_ms: int = 250
    max_segment_seconds: float = 10


@dataclass
class SttConfig:
    model: str = "small"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    pad_seconds: float = 0.3
    hallucinations: list[str] = field(default_factory=list)


@dataclass
class WakeWordConfig:
    phrases: list[str] = field(default_factory=lambda: ["하이엘지", "하이lg"])
    follow_up_seconds: float = 5


def _default_labels() -> dict[str, list[str]]:
    return {
        "SPEECH": ["Speech", "Child speech, kid speaking", "Narration, monologue", "Whispering"],
        "CONVERSATION": ["Conversation", "Chatter", "Hubbub, speech noise, speech babble"],
        "MUSIC": ["Music", "Musical instrument", "Singing", "Song"],
        "TV": ["Television"],
        "SILENT": ["Silence"],
    }


# 라벨로 정할 수 있는 값. LOUD_NOISE(소리 크기)와 UNKNOWN(판단 불가)은 규칙으로 정한다
LABEL_CONTEXTS = {"SPEECH", "CONVERSATION", "MUSIC", "TV", "SILENT"}


@dataclass
class ClassifierConfig:
    model_path: str = "data/models/yamnet.tflite"
    window_seconds: float = 1.0
    vote_windows: int = 3
    min_score: float = 0.2
    silent_below_dbfs: float = -55
    loud_above_dbfs: float = -20
    level_confidence: float = 0.8
    labels: dict[str, list[str]] = field(default_factory=_default_labels)

    def __post_init__(self):
        unknown = self.labels.keys() - LABEL_CONTEXTS
        if unknown:
            raise ValueError(f"classifier.labels: 라벨로 정할 수 없는 값 {sorted(unknown)} (가능: {sorted(LABEL_CONTEXTS)})")

    @property
    def model_file(self) -> Path:
        path = Path(self.model_path)
        return path if path.is_absolute() else ROOT / path


@dataclass
class AudioConfig:
    mic: MicConfig
    tts: TtsConfig
    vad: VadConfig
    stt: SttConfig
    wakeword: WakeWordConfig
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    @classmethod
    def default(cls) -> "AudioConfig":
        return cls(MicConfig(), TtsConfig(), VadConfig(), SttConfig(), WakeWordConfig(), ClassifierConfig())


def _build(section_cls, values: dict | None):
    values = values or {}
    unknown = values.keys() - {f.name for f in fields(section_cls)}
    if unknown:
        raise ValueError(f"{section_cls.__name__}: 알 수 없는 설정 {sorted(unknown)}")
    return section_cls(**values)


def load_audio_config(path: Path | str = DEFAULT_AUDIO_CONFIG) -> AudioConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AudioConfig(
        mic=_build(MicConfig, data.get("mic")),
        tts=_build(TtsConfig, data.get("tts")),
        vad=_build(VadConfig, data.get("vad")),
        stt=_build(SttConfig, data.get("stt")),
        wakeword=_build(WakeWordConfig, data.get("wakeword")),
        classifier=_build(ClassifierConfig, data.get("classifier")),
    )
