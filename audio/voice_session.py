"""음성 대화 상태 머신.

    IDLE ──"하이 엘지, 불 켜줘"──▶ VOICE_COMMAND "불 켜줘"
     │  └─"하이 엘지"──▶ AWAIT_COMMAND ──(follow_up_seconds 안에 말함)──▶ VOICE_COMMAND
     │
     └─ speak 수신 ─▶ 말하기(마이크 무시) ─┬─ expect_reply=False ─▶ IDLE
                                          └─ expect_reply=True ─▶ LISTEN_REPLY
                                                ├─ 말함 ─▶ VOICE_REPLY {text}
                                                └─ listen_seconds 동안 말 없음 ─▶ VOICE_REPLY {no_response}

- 웨이크워드가 없는 발화는 Node 안에서 버린다 (내용을 내보내지도, 찍지도 않음, R-10).
- speak 명령은 IDLE일 때 하나씩 처리한다 (대답을 듣는 중에는 기다림).
- 말소리 버퍼는 인식이 끝나면 바로 버린다.
- 대답을 들을 때는 예상 대답 힌트를 준 인식(transcribe_reply)을 쓰고, 듣는 동안의 소리(최대 listen_seconds)를 모아 둔다.
  말소리 구간을 하나도 못 찾았으면 마감 때 모아 둔 소리 전체를 한 번 더 인식한다.
  (2026-11-11 실측: 침대에서 작게 대답하면 webrtcvad가 말소리 구간을 놓침. Whisper 내장 VAD가 잡음에서는 지어내지 않음을 확인)
  모아 둔 소리는 대답을 보낸 뒤 바로 버린다.
"""

import math
import queue
import threading
import time
from collections.abc import Callable
from enum import Enum

import numpy as np
from pydantic import ValidationError

from audio.config import AudioConfig
from audio.mic import _FanOut
from audio.stt import Transcript
from audio.vad import SpeechSegmenter
from audio.wakeword import TextWakeWord
from publisher import topics
from publisher.sinks import Sink
from schema.models import SpeakCommand, VoiceCommand, VoiceReply


def _dbfs(samples) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) / 32768.0
    return 20 * math.log10(max(rms, 1e-9))


class State(str, Enum):
    IDLE = "IDLE"
    AWAIT_COMMAND = "AWAIT_COMMAND"
    LISTEN_REPLY = "LISTEN_REPLY"


class VoiceSession:
    def __init__(
        self,
        source: _FanOut,
        segmenter: SpeechSegmenter,
        transcribe: Callable[[object], Transcript | None],
        say: Callable[[str], None],
        wakeword: TextWakeWord,
        sink: Sink,
        node_id: str,
        room_id: str,
        config: AudioConfig,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
        debug: bool = False,
        record_dir: str | None = None,
        transcribe_reply: Callable[[object], Transcript | None] | None = None,
    ):
        self.source = source
        self.frames = source.subscribe()
        self.segmenter = segmenter
        self.transcribe = transcribe
        self.transcribe_reply = transcribe_reply or transcribe
        self._window: list = []      # 대답 듣는 동안의 소리 (대답을 보내면 버림)
        self._tried_segment = False  # 대답 듣는 동안 말소리 구간을 하나라도 인식해 봤는지
        self.say = say
        self.wakeword = wakeword
        self.sink = sink
        self.node_id = node_id
        self.room_id = room_id
        self.config = config
        self.clock = clock
        self.log = log
        self.state = State.IDLE
        self.debug = debug          # 테스트용: 대답 듣는 동안의 소리 크기·말소리 구간·인식 결과를 화면에 표시
        self._peak_dbfs = -120.0    # 대답 듣는 동안 들린 가장 큰 소리 (debug)
        # 평가용 녹음 (사용자 동의 시에만): 질문에 대한 대답을 듣는 동안의 소리만 WAV로 저장. 그 밖의 소리는 저장하지 않음
        self.record_dir = record_dir
        self._commands: queue.Queue[SpeakCommand] = queue.Queue()
        self._deadline = 0.0
        self._question: SpeakCommand | None = None

    # ---------- 입력 ----------

    def submit(self, command: SpeakCommand) -> None:
        """다른 스레드(MQTT)에서 불러도 됨."""
        self._commands.put(command)

    def on_speak_message(self, topic: str, payload: bytes) -> None:
        """MqttSink.subscribe 용 핸들러."""
        try:
            self.submit(SpeakCommand.model_validate_json(payload))
        except ValidationError as e:
            self.log(f"[음성] 잘못된 speak 메시지 무시: {e.errors()[0]['msg']}")

    # ---------- 실행 ----------

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.step(timeout=0.05)

    def step(self, timeout: float = 0.05) -> None:
        if self.state == State.IDLE and not self._commands.empty():
            self._speak(self._commands.get())
        try:
            frame = self.frames.get(timeout=timeout) if timeout > 0 else self.frames.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            if self.state == State.LISTEN_REPLY:
                self._window.append(frame)
                if self.debug:
                    self._peak_dbfs = max(self._peak_dbfs, _dbfs(frame))
            segment = self.segmenter.push(frame)
            if segment is not None:
                self._on_segment(segment)
        self._check_deadline()

    def _speak(self, command: SpeakCommand) -> None:
        self.log(f"[음성] 🔊 '{command.text}'")
        self.source.mute()
        self._drain()  # 말하기 전에 쌓인 소리는 버림
        try:
            self.say(command.text)
        finally:
            self.source.unmute(self.config.tts.mute_tail_seconds)
        if command.expect_reply:
            self.state = State.LISTEN_REPLY
            self._question = command
            self._deadline = self.clock() + self.config.tts.mute_tail_seconds + command.listen_seconds
            self._peak_dbfs = -120.0
            self._window, self._tried_segment = [], False
            self.log(f"[음성] 대답 듣는 중 (최대 {command.listen_seconds:g}초)")

    def _drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        self.segmenter.reset()

    def _recognize(self, audio, reply: bool) -> Transcript | None:
        try:
            return (self.transcribe_reply if reply else self.transcribe)(audio)
        except Exception as e:  # 인식 실패는 들은 게 없는 것으로 처리 (지어내지 않음)
            self.log(f"[음성] 인식 오류: {e}")
            return None

    def _on_segment(self, segment) -> None:
        listening = self.state == State.LISTEN_REPLY
        if listening:
            self._tried_segment = True
        result = self._recognize(segment, reply=listening)
        if self.debug and self.state == State.LISTEN_REPLY:
            heard = "인식 결과 없음 (잡음·환각으로 처리)" if result is None else f"'{result.text}' (신뢰도 {result.confidence:.2f})"
            self.log(f"[음성 테스트] 말소리 구간 {len(segment) / 16000:.2f}초, 크기 {_dbfs(segment):.0f} dBFS → {heard}")
        del segment

        if self.state == State.LISTEN_REPLY:
            if result is None:
                return  # 잡음. 마감 시간까지 계속 듣는다
            self._reply(result.text, result.confidence)
        elif self.state == State.AWAIT_COMMAND:
            if result is None:
                return
            command = self.wakeword.match(result.text)  # "하이 엘지"를 또 말했을 수도 있음
            self._command(command if command else result.text, result.confidence)
        elif result is not None:
            command = self.wakeword.match(result.text)
            if command is None:
                return  # 웨이크워드 없음 → 버림
            if command:
                self._command(command, result.confidence)
            else:
                self.state = State.AWAIT_COMMAND
                self._deadline = self.clock() + self.config.wakeword.follow_up_seconds
                self.log("[음성] '하이 엘지' 감지 → 명령 기다리는 중")

    def _save_eval(self, audio) -> None:
        import wave
        from datetime import datetime
        from pathlib import Path

        folder = Path(self.record_dir)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{str(self._question.command_id)[:8]}.wav"
        with wave.open(str(folder / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.astype("int16").tobytes())
        self.log(f"[음성 테스트] 평가용 녹음 저장: {name}")

    def _check_deadline(self) -> None:
        if self.state == State.IDLE or self.segmenter.in_speech or self.clock() < self._deadline:
            return
        if self.state == State.LISTEN_REPLY:
            result = None
            if not self._tried_segment and self._window:
                # 말소리 구간을 못 찾음 → 들은 소리 전체를 한 번 더 인식 (작은 목소리 대비)
                result = self._recognize(np.concatenate(self._window), reply=True)
                if self.debug:
                    heard = "인식 결과 없음" if result is None else f"'{result.text}' (신뢰도 {result.confidence:.2f})"
                    self.log(f"[음성 테스트] 말소리 구간을 못 찾아 전체 {len(self._window) * 0.03:.1f}초를 다시 인식 "
                             f"(가장 큰 소리 {self._peak_dbfs:.0f} dBFS) → {heard}")
            if result is None:
                self._reply(None, None)
            else:
                self._reply(result.text, result.confidence)
        else:
            self.log("[음성] 명령 없음 → 대기")
            self.state = State.IDLE

    # ---------- 출력 ----------

    def _reply(self, text: str | None, confidence: float | None) -> None:
        if self.record_dir and self._window:
            self._save_eval(np.concatenate(self._window))
        self._window = []  # 들은 소리는 대답을 보낸 뒤 바로 버림
        reply = VoiceReply(
            node_id=self.node_id,
            room_id=self.room_id,
            command_id=self._question.command_id,
            text=text,
            confidence=confidence,
            no_response=text is None,
        )
        self.sink.publish(topics.voice_reply(self.room_id), reply)
        self.state, self._question = State.IDLE, None

    def _command(self, text: str, confidence: float) -> None:
        command = VoiceCommand(node_id=self.node_id, room_id=self.room_id, text=text, confidence=confidence)
        self.sink.publish(topics.voice_command(self.room_id), command)
        self.state = State.IDLE
