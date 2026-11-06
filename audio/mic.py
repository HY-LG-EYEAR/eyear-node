"""마이크 입력. 스트림 하나를 여러 소비자(소리 분류, 말소리 검출)에 나눠 준다.

- 16kHz 모노, 30ms(480샘플) int16 프레임 단위 (webrtcvad가 받는 형식)
- mute 중에는 프레임을 아무에게도 주지 않는다 (Node가 자기 스피커 소리를 듣지 않게)
- 소리는 메모리에서만 처리하고 저장하지 않는다 (R-10)

WavSource는 같은 인터페이스로 WAV 파일을 읽는다 (테스트·재생용).
"""

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # 30ms
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class _FanOut:
    def __init__(self):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._muted_until = 0.0
        self._muted = False

    def subscribe(self, maxsize: int = 500) -> queue.Queue:
        """프레임을 받을 큐. 소비자가 느려서 가득 차면 오래된 프레임부터 버린다."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def mute(self) -> None:
        self._muted = True

    def unmute(self, tail_seconds: float = 0.0) -> None:
        self._muted_until = time.monotonic() + tail_seconds
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted or time.monotonic() < self._muted_until

    def _deliver(self, frame: np.ndarray) -> None:
        if self.muted:
            return
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(frame)


class MicStream(_FanOut):
    def __init__(self, device: int | str | None = None):
        import sounddevice as sd  # 실제로 쓸 때만 불러옴

        super().__init__()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        self._deliver(indata[:, 0].copy())

    def start(self) -> "MicStream":
        self._stream.start()
        return self

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class WavSource(_FanOut):
    """16kHz 모노 16비트 WAV를 30ms 프레임으로 흘려보낸다. realtime=False면 최대한 빨리."""

    def __init__(self, path: Path | str, realtime: bool = True, trailing_silence: float = 1.5):
        super().__init__()
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"{path}: 16kHz 모노 16비트 WAV만 지원")
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        # 파일 끝에 무음을 붙여 마지막 말소리 구간이 끝나게 함
        samples = np.concatenate([samples, np.zeros(int(trailing_silence * SAMPLE_RATE), np.int16)])
        usable = len(samples) // FRAME_SAMPLES * FRAME_SAMPLES
        self._frames = samples[:usable].reshape(-1, FRAME_SAMPLES)
        self._realtime = realtime
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.finished = threading.Event()

    def _run(self) -> None:
        for frame in self._frames:
            self._deliver(frame)
            if self._realtime:
                time.sleep(FRAME_SECONDS)
        self.finished.set()

    def start(self) -> "WavSource":
        self._thread.start()
        return self

    def close(self) -> None:
        pass
