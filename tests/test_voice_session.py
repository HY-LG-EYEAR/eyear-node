"""음성 대화 상태 머신 테스트 (가짜 마이크·인식·TTS, 가짜 시계)."""

import numpy as np
import pytest

from audio.config import AudioConfig
from audio.mic import FRAME_SAMPLES, FRAME_SECONDS, _FanOut
from audio.stt import Transcript
from audio.vad import SpeechSegmenter
from audio.voice_session import State, VoiceSession
from audio.wakeword import TextWakeWord
from schema.models import SpeakCommand, VoiceCommand, VoiceReply

WAKE = TextWakeWord(["하이엘지", "하이lg", "highlg", "하이예지"])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("하이 LG 불 켜줘", "불 켜줘"),
        ("high lg 불 켜줘", "불 켜줘"),
        ("하이 엘지, 불 켜줘.", "불 켜줘."),
        ("Hi... 하이예지!", ""),
        ("하이 LG", ""),
        ("음 하이 엘지 에어컨 꺼줘", "에어컨 꺼줘"),
        ("오늘 날씨 좋다", None),
        ("엘지 하이", None),
    ],
)
def test_wakeword(text, expected):
    assert WAKE.match(text) == expected


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, topic, message):
        self.messages.append((topic, message))

    def close(self):
        pass


class Harness:
    """프레임 값 k(1 이상)는 말소리, 그 구간은 texts[k]로 인식된다고 가정."""

    def __init__(self, texts: dict[int, str | None]):
        self.config = AudioConfig.default()
        self.config.tts.mute_tail_seconds = 0
        self.clock = Clock()
        self.source = _FanOut()
        self.sink = Recorder()
        self.said = []
        self.texts = texts
        self.session = VoiceSession(
            source=self.source,
            segmenter=SpeechSegmenter(self.config.vad, is_speech=lambda f: bool(f.any())),
            transcribe=self._transcribe,
            say=self._say,
            wakeword=WAKE,
            sink=self.sink,
            node_id="node_room1",
            room_id="room1",
            config=self.config,
            clock=self.clock,
            log=lambda _: None,
        )

    def _transcribe(self, segment):
        k = int(segment.max())
        return None if self.texts[k] is None else Transcript(self.texts[k], 0.6)

    def _say(self, text):
        assert self.source.muted, "말하는 동안 마이크가 켜져 있음"
        self.said.append(text)

    def frames(self, value: int, seconds: float):
        for _ in range(round(seconds / FRAME_SECONDS)):
            self.source._deliver(np.full(FRAME_SAMPLES, value, np.int16))
            self.clock.t += FRAME_SECONDS
            self.session.step(timeout=0)

    def talk(self, k: int, seconds: float = 0.6):
        self.frames(k, seconds)
        self.frames(0, 1.0)  # 말 끝 (0.8초 무음) 판단까지

    def idle(self, seconds: float):
        for _ in range(round(seconds / FRAME_SECONDS)):
            self.clock.t += FRAME_SECONDS
            self.session.step(timeout=0)

    def published(self):
        return [m for _, m in self.sink.messages]


def test_wake_and_command_in_one_utterance():
    h = Harness({1: "하이 LG 불 켜줘"})
    h.talk(1)
    [(topic, msg)] = h.sink.messages
    assert topic == "eyear/v0/room1/voice/command"
    assert isinstance(msg, VoiceCommand) and msg.text == "불 켜줘"


def test_wake_then_command_in_next_utterance():
    h = Harness({1: "하이 LG", 2: "에어컨 꺼줘"})
    h.talk(1)
    assert h.session.state == State.AWAIT_COMMAND and h.published() == []
    h.talk(2)
    assert [m.text for m in h.published()] == ["에어컨 꺼줘"]
    assert h.session.state == State.IDLE


def test_wake_without_command_times_out():
    h = Harness({1: "하이 LG"})
    h.talk(1)
    h.idle(h.config.wakeword.follow_up_seconds + 0.1)
    assert h.session.state == State.IDLE and h.published() == []


def test_speech_without_wakeword_is_discarded():
    h = Harness({1: "오늘 저녁 뭐 먹지", 2: None})
    h.talk(1)
    h.talk(2)
    assert h.published() == []


def test_question_and_reply():
    h = Harness({1: "응?"})
    question = SpeakCommand(text="A님, 주무실 건가요?", expect_reply=True, listen_seconds=5)
    h.session.submit(question)
    h.idle(0.1)
    assert h.said == ["A님, 주무실 건가요?"] and h.session.state == State.LISTEN_REPLY
    h.talk(1)
    [(topic, reply)] = h.sink.messages
    assert topic == "eyear/v0/room1/voice/reply"
    assert isinstance(reply, VoiceReply)
    assert (reply.command_id, reply.text, reply.no_response) == (question.command_id, "응?", False)


def test_reply_while_listening_is_not_treated_as_wakeword():
    h = Harness({1: "하이 LG 불 꺼줘"})
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True))
    h.idle(0.1)
    h.talk(1)
    [reply] = h.published()
    assert isinstance(reply, VoiceReply) and reply.text == "하이 LG 불 꺼줘"


def test_no_response_after_listen_seconds():
    h = Harness({})
    question = SpeakCommand(text="주무실 건가요?", expect_reply=True, listen_seconds=5)
    h.session.submit(question)
    h.idle(4.9)
    assert h.published() == []
    h.idle(0.2)
    [reply] = h.published()
    assert (reply.command_id, reply.text, reply.confidence, reply.no_response) == (question.command_id, None, None, True)


def test_noise_while_listening_keeps_waiting():
    h = Harness({1: None, 2: "아니"})
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True, listen_seconds=5))
    h.idle(0.1)
    h.talk(1)  # 기침 → 인식 결과 없음
    assert h.published() == [] and h.session.state == State.LISTEN_REPLY
    h.talk(2)
    assert [m.text for m in h.published()] == ["아니"]


def test_speech_in_progress_at_deadline_is_waited_for():
    h = Harness({1: "응"})
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True, listen_seconds=1))
    h.idle(0.5)
    h.talk(1, seconds=1.5)  # 마감(1초)을 넘겨서까지 말함
    assert [m.text for m in h.published()] == ["응"]


def test_announcement_does_not_listen_and_commands_run_in_order():
    h = Harness({})
    h.session.submit(SpeakCommand(text="네, 불을 켤게요", expect_reply=False))
    h.session.submit(SpeakCommand(text="두 번째", expect_reply=False))
    h.idle(0.1)
    assert h.said == ["네, 불을 켤게요", "두 번째"] and h.session.state == State.IDLE
    assert h.published() == []


def test_speak_waits_until_reply_finished():
    h = Harness({1: "응"})
    h.session.submit(SpeakCommand(text="질문", expect_reply=True))
    h.idle(0.1)
    h.session.submit(SpeakCommand(text="안내", expect_reply=False))
    h.idle(0.1)
    assert h.said == ["질문"]  # 대답 듣는 중에는 다음 문장을 말하지 않음
    h.talk(1)
    h.idle(0.1)
    assert h.said == ["질문", "안내"]


def test_invalid_speak_message_is_ignored():
    h = Harness({})
    h.session.on_speak_message("eyear/v0/room1/speak", b'{"text": ""}')
    h.idle(0.1)
    assert h.said == []


class QuietHarness(Harness):
    """VAD가 작은 목소리를 놓치는 상황: segmenter는 아무것도 말소리로 보지 않는다."""

    def __init__(self, reply_text):
        super().__init__({})
        self.reply_calls = []
        self.session.segmenter = SpeechSegmenter(self.config.vad, is_speech=lambda f: False)
        self.session.transcribe_reply = self._reply
        self.reply_text = reply_text

    def _reply(self, audio):
        self.reply_calls.append(len(audio))
        return None if self.reply_text is None else Transcript(self.reply_text, 0.3)


def test_quiet_reply_missed_by_vad_is_found_in_whole_window():
    h = QuietHarness("응")
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True, listen_seconds=5))
    h.idle(0.1)
    h.frames(3, 5.5)  # 작은 목소리 (VAD는 못 잡음)
    [reply] = h.published()
    assert (reply.text, reply.no_response) == ("응", False)
    assert len(h.reply_calls) == 1 and h.reply_calls[0] > 16000 * 4  # 들은 소리 전체를 한 번 인식
    assert h.session._window == []  # 보낸 뒤 버림


def test_whole_window_noise_is_still_no_response():
    h = QuietHarness(None)
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True, listen_seconds=5))
    h.idle(0.1)
    h.frames(3, 5.5)
    [reply] = h.published()
    assert reply.no_response and reply.text is None


def test_reply_uses_hinted_recognizer_but_commands_do_not():
    h = Harness({1: "하이 LG 불 켜줘", 2: "응"})
    used = []
    h.session.transcribe_reply = lambda seg: (used.append("reply"), Transcript("응", 0.5))[1]
    h.talk(1)  # 명령 → 일반 인식
    assert used == [] and h.published()[0].text == "불 켜줘"
    h.session.submit(SpeakCommand(text="주무실 건가요?", expect_reply=True))
    h.idle(0.1)
    h.talk(2)  # 대답 → 힌트 인식
    assert used == ["reply"] and h.published()[-1].text == "응"
