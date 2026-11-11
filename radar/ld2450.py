"""HLK-LD2450 (24GHz, 최대 3명 위치 추적) 보고 프레임 파서.

근거: Hi-Link "HLK-LD2450 Serial Communication Protocol" V1.03 (2023-10-17), 2.3절 표 9·10.
- 직렬: 256000 baud, 1 stop bit, no parity. 초당 10 프레임
- 프레임 (30바이트): 헤더 AA FF 03 00 | 목표 1·2·3 각 8바이트 | 끝 55 CC
- 목표 8바이트 (리틀 엔디언 uint16 ×4): X(mm), Y(mm), 속도(cm/s), 거리 해상도(mm)
- X·Y·속도의 부호: 최상위 비트 1 = 양수, 0 = 음수, 나머지 15비트가 크기 (2의 보수가 아님)
  문서 예: X 0x030E → 최상위 비트 0 → -782 mm, Y 0x86B1 → 34481 - 2^15 = 1713 mm
- 목표가 없으면 그 8바이트는 모두 0
좌표: 레이더 기준. Y = 레이더 정면 방향 거리, X = 좌우 (좌우 방향은 설치 후 실측으로 확인할 것).
"""

import struct
from dataclasses import dataclass

HEADER = bytes.fromhex("AAFF0300")
FOOTER = bytes.fromhex("55CC")
FRAME_LEN = 30
TARGET_LEN = 8
BAUD = 256000


@dataclass(frozen=True)
class Ld2450Target:
    x_mm: int
    y_mm: int
    speed_cms: int
    resolution_mm: int


def decode_signed(raw: int) -> int:
    """최상위 비트 1 = 양수, 0 = 음수 (문서 표 10)."""
    return raw - 0x8000 if raw & 0x8000 else -raw


def encode_signed(value: int) -> int:
    return value + 0x8000 if value >= 0 else -value


def parse_frame(frame: bytes) -> list[Ld2450Target]:
    """30바이트 프레임 → 있는 목표들. 형식이 틀리면 ValueError."""
    if len(frame) != FRAME_LEN or frame[:4] != HEADER or frame[-2:] != FOOTER:
        raise ValueError("LD2450 프레임 형식이 아님")
    targets = []
    for i in range(3):
        chunk = frame[4 + i * TARGET_LEN: 4 + (i + 1) * TARGET_LEN]
        if not any(chunk):
            continue  # 목표 없음
        x, y, speed, resolution = struct.unpack("<HHHH", chunk)
        targets.append(Ld2450Target(decode_signed(x), decode_signed(y), decode_signed(speed), resolution))
    return targets


def build_frame(targets: list[Ld2450Target]) -> bytes:
    """mock·테스트용: 목표(최대 3개) → 30바이트 프레임."""
    if len(targets) > 3:
        raise ValueError("LD2450은 최대 3명")
    body = b"".join(
        struct.pack("<HHHH", encode_signed(t.x_mm), encode_signed(t.y_mm), encode_signed(t.speed_cms), t.resolution_mm)
        for t in targets
    )
    return HEADER + body.ljust(3 * TARGET_LEN, b"\x00") + FOOTER


class Ld2450StreamParser:
    """직렬로 들어오는 바이트 조각을 받아 완성된 프레임마다 목표 목록을 돌려준다.

    중간부터 읽기 시작하거나 바이트가 깨져도 다음 헤더를 찾아 다시 맞춘다.
    """

    def __init__(self):
        self._buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[list[Ld2450Target]]:
        self._buf += data
        frames = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                del self._buf[: max(0, len(self._buf) - len(HEADER) + 1)]  # 헤더 일부일 수 있는 끝만 남김
                return frames
            if len(self._buf) - start < FRAME_LEN:
                del self._buf[:start]
                return frames
            frame = bytes(self._buf[start: start + FRAME_LEN])
            if frame[-2:] == FOOTER:
                frames.append(parse_frame(frame))
                del self._buf[: start + FRAME_LEN]
            else:
                self.bad_frames += 1
                del self._buf[: start + 1]  # 가짜 헤더였음 → 다음 위치부터 다시 찾기
