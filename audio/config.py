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


@dataclass
class AudioConfig:
    mic: MicConfig
    tts: TtsConfig
    vad: VadConfig
    stt: SttConfig
    wakeword: WakeWordConfig

    @classmethod
    def default(cls) -> "AudioConfig":
        return cls(MicConfig(), TtsConfig(), VadConfig(), SttConfig(), WakeWordConfig())


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
    )
