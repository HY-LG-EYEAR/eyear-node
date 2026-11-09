"""소리 분류 규칙, 다수결, 방 상태 합치기 테스트 (마이크·모델 없이)."""

import numpy as np
import pytest

from audio.classifier import AudioContextWorker, ContextVoter, classify_window, dbfs
from audio.config import ClassifierConfig, load_audio_config
from audio.mic import FRAME_SAMPLES, _FanOut
from fusion.room_state import RoomState
from schema.enums import AudioContext as A
from schema.enums import Light
from schema.family import load_family
from schema.models import Audio
from vision.tracker import VisionPerson

CFG = ClassifierConfig()  # min_score 0.2, SILENT < -55 dBFS, LOUD > -20 dBFS
NORMAL, QUIET, LOUD = -35.0, -70.0, -10.0
FAMILY = load_family()


@pytest.mark.parametrize(
    "scores, level, expected",
    [
        ({"Speech": 0.9}, QUIET, A.SILENT),                       # 소리 크기가 먼저
        ({"Speech": 0.8, "Music": 0.3}, NORMAL, A.SPEECH),
        ({"Speech": 0.4, "Conversation": 0.6}, NORMAL, A.CONVERSATION),
        ({"Music": 0.7, "Speech": 0.3}, NORMAL, A.MUSIC),
        ({"Singing": 0.5}, NORMAL, A.MUSIC),                      # 묶음 안의 다른 라벨
        ({"Television": 0.5, "Speech": 0.4}, NORMAL, A.TV),
        ({"Speech": 0.6}, LOUD, A.SPEECH),                        # 큰 말소리는 말소리
        ({"Vacuum cleaner": 0.9}, LOUD, A.LOUD_NOISE),
        ({"Silence": 0.6}, NORMAL, A.SILENT),
        ({"Typing": 0.8}, NORMAL, A.UNKNOWN),                     # 분류할 수 없는 소리는 지어내지 않음
        ({"Speech": 0.1}, NORMAL, A.UNKNOWN),                     # 점수가 낮으면 말소리로 보지 않음
    ],
)
def test_classify_window(scores, level, expected):
    context, confidence = classify_window(scores, level, CFG)
    assert context == expected
    assert (confidence == 0.0) == (expected == A.UNKNOWN)


def test_dbfs():
    assert dbfs(np.zeros(100, np.int16)) < -150
    assert dbfs(np.full(100, 32767, np.int16)) == pytest.approx(0, abs=0.01)


def test_voter_majority_tie_and_confidence():
    v = ContextVoter(3)
    assert v.push(A.SPEECH, 0.8).context == "SPEECH"
    assert v.push(A.MUSIC, 0.6).context == "MUSIC"            # 1:1 동점 → 최근 값
    out = v.push(A.SPEECH, 0.6)
    assert (out.context, out.confidence) == ("SPEECH", 0.7)  # 2:1, 이긴 값 평균
    assert v.push(A.MUSIC, 0.5).context == "MUSIC"            # 첫 SPEECH가 밀려남 → MUSIC 2:1


def test_config_file_labels_are_valid():
    config = load_audio_config().classifier
    assert set(config.labels) <= {"SPEECH", "CONVERSATION", "MUSIC", "TV", "SILENT"}
    with pytest.raises(ValueError):
        ClassifierConfig(labels={"LOUD_NOISE": ["Bang"]})


class FakeScores:
    def __init__(self, scores):
        self.scores, self.calls = scores, 0

    def __call__(self, samples):
        self.calls += 1
        assert len(samples) == round(CFG.window_seconds * 16000 / FRAME_SAMPLES) * FRAME_SAMPLES
        return self.scores


def test_worker_classifies_each_second_and_holds_while_muted():
    source = _FanOut()
    results = []
    model = FakeScores({"Speech": 0.9})
    worker = AudioContextWorker(source, model, CFG, results.append)
    loud_enough = np.full(FRAME_SAMPLES, 1000, np.int16)  # 약 -30 dBFS

    for _ in range(33):  # 30ms × 33 ≈ 1초
        source._deliver(loud_enough)
        worker.step(timeout=0)
    assert model.calls == 1 and [r.context for r in results] == ["SPEECH"]

    source.mute()  # Node가 말하는 중 → 프레임이 오지 않음 → 직전 값 유지
    worker.step(timeout=0)
    assert model.calls == 1 and results[-1].context == "SPEECH" and len(results) == 2

    source.unmute()
    worker.step(timeout=0)  # 마이크가 켜져 있는데 소리가 안 옴 → 결과를 내지 않음 (방 상태에서 stale → UNKNOWN)
    assert len(results) == 2


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def person(identity="MEMBER_A"):
    return VisionPerson(track_id=1, bbox=(0, 0, 10, 10), identity=identity, identity_confidence=0.9,
                        posture="STANDING", posture_confidence=0.8)


def test_room_state_merges_camera_and_audio():
    clock = Clock()
    room = RoomState(FAMILY, stale_seconds=3, clock=clock)
    snap = room.snapshot()  # 아직 아무 결과 없음 → 모두 UNKNOWN
    assert (snap.occupancy.state, snap.persons, snap.light, snap.audio.context) == ("UNKNOWN", [], "UNKNOWN", "UNKNOWN")

    room.set_camera([person()], Light.BRIGHT)
    room.set_audio(Audio(context="MUSIC", confidence=0.7))
    snap = room.snapshot()
    assert (snap.occupancy.state, snap.persons[0].identity, snap.light) == ("OCCUPIED", "MEMBER_A", "BRIGHT")
    assert (snap.audio.context, snap.audio.confidence) == ("MUSIC", 0.7)
    assert snap.persons[0].identity_source == "FACE" and snap.persons[0].position is None


def test_room_state_stale_sensor_becomes_unknown():
    clock = Clock()
    room = RoomState(FAMILY, stale_seconds=3, clock=clock)
    room.set_camera([], Light.DIM)
    room.set_audio(Audio(context="SILENT", confidence=0.8))
    clock.t += 2
    room.set_camera([], Light.DIM)  # 카메라만 계속 갱신
    clock.t += 2                    # 소리는 4초째 소식 없음
    snap = room.snapshot()
    assert (snap.occupancy.state, snap.occupancy.confidence, snap.light) == ("EMPTY", 0.5, "DIM")
    assert snap.audio.context == "UNKNOWN"
    clock.t += 2
    assert room.snapshot().occupancy.state == "UNKNOWN"


def test_room_state_camera_unreadable():
    room = RoomState(FAMILY, clock=Clock())
    room.set_camera(None, Light.UNKNOWN)
    snap = room.snapshot()
    assert (snap.occupancy.state, snap.light) == ("UNKNOWN", "UNKNOWN")


@pytest.mark.slow
def test_real_yamnet_on_tts_speech_and_silence(tmp_path):
    """macOS TTS 말소리 → SPEECH, 무음 → SILENT. pytest -m slow 로 실행."""
    from audio.classifier import YamnetModel
    from audio.mic import WavSource
    from tests.test_audio import tts_wav

    config = load_audio_config().classifier
    model = YamnetModel(config)
    source = WavSource(tts_wav(tmp_path, "오늘 저녁에는 거실에서 같이 영화를 볼까요"), realtime=False, trailing_silence=0)
    samples = np.concatenate(list(source._frames))[4000:20000]
    context, _ = classify_window(model.scores(samples), dbfs(samples), config)
    assert context == A.SPEECH
    silence = np.zeros(16000, np.int16)
    assert classify_window(model.scores(silence), dbfs(silence), config)[0] == A.SILENT
    model.close()
