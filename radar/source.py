"""레이더 바이트 공급원: 실제 시리얼 포트 또는 같은 바이트를 만드는 mock.

둘 다 별도 스레드에서 바이트 조각을 on_bytes로 넘긴다. 파서는 공급원이 무엇인지 모른다 (R-05).
실물이 오면 mock 대신 시리얼 포트 이름만 넘기면 된다.
"""

import math
import threading
import time
from collections.abc import Callable

from radar import ld2410c, ld2450
from radar.ld2410c import Ld2410Report, TargetState
from radar.ld2450 import Ld2450Target


def list_ports() -> list[str]:
    from serial.tools import list_ports as lp

    return [f"{p.device}  ({p.description})" for p in lp.comports()]


class SerialSource:
    def __init__(self, port: str, on_bytes: Callable[[bytes], None], baud: int = 256000):
        import serial  # pyserial, 실제로 쓸 때만 불러옴

        self._serial = serial.Serial(port, baud, timeout=0.1)
        self._on_bytes = on_bytes
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"serial {port}")

    def _run(self) -> None:
        while not self._stop.is_set():
            data = self._serial.read(self._serial.in_waiting or 1)
            if data:
                self._on_bytes(data)

    def start(self) -> "SerialSource":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self._serial.close()


class MockSource:
    """mock: 시간 t → 프레임 바이트를 frame_hz로 만들어 실제 포트처럼 조각내 보낸다.

    기본 시나리오 (반복): 0~5초 없음 → 5~10초 문에서 침대로 걸어옴 → 10초~ 누워서 가만히.
    LD2450은 가만히 있으면 still_lost_after초 뒤 목표를 놓친다고 가정 (실측 전 가정값).
    """

    def __init__(self, kind: str, on_bytes: Callable[[bytes], None], frame_hz: float = 10,
                 period: float = 40.0, still_lost_after: float = 5.0):
        self.kind = kind
        self._on_bytes = on_bytes
        self._interval = 1.0 / frame_hz
        self.period = period
        self.still_lost_after = still_lost_after
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"mock {kind}")

    def frame_at(self, t: float) -> bytes:
        t = t % self.period
        if self.kind == "ld2450":
            return ld2450.build_frame(self._ld2450_targets(t))
        return ld2410c.build_frame(self._ld2410_report(t))

    def _ld2450_targets(self, t: float) -> list[Ld2450Target]:
        if t < 5:
            return []
        if t < 10:  # 문(-1.5m, 0.5m) → 침대(1.0m, 2.5m)를 걸어옴
            k = (t - 5) / 5
            return [Ld2450Target(int(-1500 + 2500 * k), int(500 + 2000 * k), 50, 320)]
        if t - 10 < self.still_lost_after:  # 누운 뒤 잠깐은 잡힘 (미세한 움직임)
            return [Ld2450Target(1000 + int(20 * math.sin(t)), 2500, 0, 320)]
        return []  # 가만히 누운 사람을 놓침

    @staticmethod
    def _ld2410_report(t: float) -> Ld2410Report:
        if t < 5:
            return Ld2410Report(TargetState.NONE, 0, 0, 0, 0, 0)
        if t < 10:
            return Ld2410Report(TargetState.MOVING_AND_STATIONARY, 250, 70, 250, 30, 250)
        return Ld2410Report(TargetState.STATIONARY, 0, 0, 270, 45, 270)

    def _run(self) -> None:
        start = time.monotonic()
        while not self._stop.is_set():
            frame = self.frame_at(time.monotonic() - start)
            half = len(frame) // 2  # 실제 시리얼처럼 조각내서 보냄
            self._on_bytes(frame[:half])
            self._on_bytes(frame[half:])
            time.sleep(self._interval)

    def start(self) -> "MockSource":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
