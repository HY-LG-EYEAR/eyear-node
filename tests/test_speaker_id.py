"""화자 식별 (audio/speaker_id.py) 테스트. 빠른 테스트는 가짜 특징값, 느린 테스트는 실제 모델 + macOS 합성 음성."""

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from audio.enroll_voice import trim
from audio.speaker_id import ROOT, SpeakerIdentifier, VoiceGallery

SR = 16000


def _unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


VOICES = {1: _unit(1, 0, 0), 2: _unit(0, 1, 0), 3: _unit(0.7, 0.7, 0.1)}


def fake_embed(audio):
    """소리 값 k → 사람 k의 특징값 (테스트용)."""
    return VOICES[int(np.max(audio))]


def gallery_with(tmp_path, people):
    g = VoiceGallery(tmp_path)
    for name, k in people.items():
        g.save(name, np.stack([VOICES[k], VOICES[k]]))
    return VoiceGallery(tmp_path)   # 파일에서 다시 읽기


def clip(k, seconds=1.5):
    return np.full(int(SR * seconds), k, np.int16)


def test_identifies_enrolled_speaker(tmp_path):
    ident = SpeakerIdentifier(fake_embed, gallery_with(tmp_path, {"MEMBER_A": 1, "MEMBER_D": 2}))
    assert ident.identify(clip(2)) == ("MEMBER_D", 1.0)
    assert ident.identify(clip(1)) == ("MEMBER_A", 1.0)


def test_short_speech_is_not_judged(tmp_path):
    ident = SpeakerIdentifier(fake_embed, gallery_with(tmp_path, {"MEMBER_D": 2}), min_seconds=1.0)
    assert ident.identify(clip(2, seconds=0.5)) == ("UNKNOWN", 0.0)   # "응"처럼 짧은 말


def test_ambiguous_or_unenrolled_voice_is_unknown(tmp_path):
    # 사람 3은 1과 2의 중간 → 1등·2등 차이가 margin보다 작음
    ident = SpeakerIdentifier(fake_embed, gallery_with(tmp_path / "a", {"MEMBER_A": 1, "MEMBER_D": 2}), threshold=0.5)
    assert ident.identify(clip(3)) == ("UNKNOWN", 0.0)
    # 등록되지 않은 목소리 (유사도가 threshold보다 낮음)
    ident = SpeakerIdentifier(fake_embed, gallery_with(tmp_path / "b", {"MEMBER_A": 1}), threshold=0.9)
    assert ident.identify(clip(2)) == ("UNKNOWN", 0.0)


def test_no_enrollment_means_unknown(tmp_path):
    assert SpeakerIdentifier(fake_embed, VoiceGallery(tmp_path / "empty")).identify(clip(1)) == ("UNKNOWN", 0.0)


def test_gallery_append_and_delete(tmp_path):
    g = gallery_with(tmp_path, {"MEMBER_D": 2})
    assert g.save("MEMBER_D", np.stack([VOICES[2]]), append=True) == 3
    assert g.delete("MEMBER_D") and not (tmp_path / "MEMBER_D.npy").exists()


def test_trim_removes_silence_around_speech():
    audio = np.concatenate([np.zeros(SR, np.int16), np.full(SR, 3000, np.int16), np.zeros(SR, np.int16)])
    assert abs(len(trim(audio)) - SR) <= 480


MODEL = ROOT / "data/models/speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


@pytest.mark.slow
@pytest.mark.skipif(not MODEL.exists() or shutil.which("say") is None, reason="모델 파일 또는 macOS say 없음")
def test_real_model_tells_synthetic_voices_apart(tmp_path):
    from audio.speaker_id import OnnxSpeakerEmbedder

    def say(voice, text, name):
        out = tmp_path / f"{name}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text], check=True)
        with wave.open(str(out)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    embed = OnnxSpeakerEmbedder(MODEL)
    enroll = ["What should we eat for dinner tonight", "It is a little hot in this room", "Please wake me up at seven"]
    voices = ["Samantha", "Daniel", "Karen", "Moira"]
    g = VoiceGallery(tmp_path / "voices")
    for v in voices:
        g.save(v, np.stack([embed(say(v, t, f"{v}{i}")) for i, t in enumerate(enroll)]))
    ident = SpeakerIdentifier(embed, VoiceGallery(tmp_path / "voices"))
    for v in voices:
        who, conf = ident.identify(say(v, "Hi LG, turn on the light", f"{v}_cmd"))
        assert who == v and conf >= 0.65
