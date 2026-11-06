"""마이크·모델 없이 확인하는 audio 테스트."""

import shutil
import subprocess
import wave

import numpy as np
import pytest

from audio.config import AudioConfig, load_audio_config
from audio.mic import FRAME_SAMPLES, SAMPLE_RATE, WavSource
from audio.stt import is_hallucination, normalize, segment_confidence
from audio.vad import FRAME_MS, SpeechSegmenter

CFG = AudioConfig.default()
SPEECH, SILENCE = np.ones(FRAME_SAMPLES, np.int16), np.zeros(FRAME_SAMPLES, np.int16)


def _segmenter():
    # 진짜 VAD 대신 "0이 아니면 말소리"
    return SpeechSegmenter(CFG.vad, is_speech=lambda f: bool(f.any()))


def _push(seg, frames):
    return [out for f in frames if (out := seg.push(f)) is not None]


def test_segment_includes_pre_roll_and_ends_after_silence():
    seg = _segmenter()
    end_frames = CFG.vad.end_silence_ms // FRAME_MS
    frames = [SILENCE] * 20 + [SPEECH] * 30 + [SILENCE] * end_frames
    out = _push(seg, frames)
    assert len(out) == 1
    pre_roll = CFG.vad.pre_roll_ms // FRAME_MS
    # 앞 무음 일부(pre-roll) + 말소리 + 끝 무음
    assert len(out[0]) == (pre_roll + 30 - CFG.vad.start_frames + end_frames) * FRAME_SAMPLES
    assert not seg.in_speech


def test_short_noise_is_dropped():
    seg = _segmenter()
    frames = [SPEECH] * CFG.vad.start_frames + [SILENCE] * 40  # 90ms뿐 → min_speech_ms 미만
    assert _push(seg, frames) == []


def test_blips_do_not_start_speech():
    seg = _segmenter()
    frames = ([SPEECH] * (CFG.vad.start_frames - 1) + [SILENCE]) * 20
    assert _push(seg, frames) == [] and not seg.in_speech


def test_long_speech_is_cut():
    seg = _segmenter()
    max_frames = int(CFG.vad.max_segment_seconds * 1000 // FRAME_MS)
    out = _push(seg, [SPEECH] * (max_frames + 10))
    assert len(out) == 1


def test_stt_helpers():
    assert normalize("하이, 엘지!  불 켜줘.") == "하이엘지불켜줘"
    assert is_hallucination("시청해 주셔서 감사합니다.", load_audio_config().stt.hallucinations)
    assert not is_hallucination("응", load_audio_config().stt.hallucinations)
    assert segment_confidence(0.0, 0.0) == 1.0
    assert segment_confidence(-10.0, 0.0) < 0.001
    assert segment_confidence(0.0, 1.0) == 0.0


def test_wav_source(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SAMPLE_RATE)
        w.writeframes(np.arange(FRAME_SAMPLES * 3, dtype=np.int16).tobytes())
    source = WavSource(path, realtime=False, trailing_silence=0)
    q = source.subscribe()
    source.mute()
    source.start()
    source.finished.wait(2)
    assert q.empty()  # mute 중에는 아무것도 전달하지 않음

    source = WavSource(path, realtime=False, trailing_silence=0)
    q = source.subscribe()
    source.start()
    source.finished.wait(2)
    assert q.qsize() == 3


def tts_wav(tmp_path, text: str):
    if shutil.which("say") is None:
        pytest.skip("macOS say 없음")
    path = tmp_path / "speech.wav"
    subprocess.run(
        ["say", "-v", load_audio_config().tts.voice, "-o", str(path),
         "--file-format=WAVE", "--data-format=LEI16@16000", text],
        check=True,
    )
    return path


@pytest.mark.slow
def test_real_vad_and_whisper_on_tts_speech(tmp_path):
    """macOS TTS로 만든 "아니"를 실제 VAD + Whisper로 인식. pytest -m slow 로 실행."""
    from audio.stt import SpeechRecognizer

    config = load_audio_config()
    source = WavSource(tts_wav(tmp_path, "아니"), realtime=False)
    q = source.subscribe(maxsize=10_000)
    source.start()
    source.finished.wait(5)
    seg = SpeechSegmenter(config.vad)
    segments = [out for _ in range(q.qsize()) if (out := seg.push(q.get())) is not None]
    assert len(segments) == 1
    result = SpeechRecognizer(config.stt).transcribe(segments[0])
    assert result is not None and normalize(result.text) == "아니"
