"""HLK-LD2410C (24GHz, 사람 재실: 움직임/정지) 보고 프레임 파서.

근거: Hi-Link "LD2410C Serial communication protocol" V1.00 (2022-11-07), 2.3절 표 9~14.
- 직렬: 256000 baud, 1 stop bit, no parity. 리틀 엔디언
- 프레임: 헤더 F4 F3 F2 F1 | 데이터 길이(2바이트) | 데이터 | 끝 F8 F7 F6 F5
- 데이터: 종류(1바이트: 0x01 엔지니어링 모드, 0x02 기본) | 0xAA | 목표 정보 | 0x55 | 0x00
- 기본 목표 정보 9바이트: 상태(1) | 움직임 목표 거리 cm(2) | 움직임 에너지(1) | 정지 목표 거리 cm(2)
  | 정지 에너지(1) | 감지 거리 cm(2)
- 상태: 0x00 없음, 0x01 움직임, 0x02 정지, 0x03 움직임+정지
- 엔지니어링 모드는 기본 정보 뒤에: 최대 움직임 게이트 N(1) | 최대 정지 게이트 N(1)
  | 게이트 0~N 움직임 에너지 | 게이트 0~N 정지 에너지 | 추가 데이터
문서 예 (기본): F4 F3 F2 F1 0D 00 02 AA 02 51 00 00 00 00 3B 00 00 55 00 F8 F7 F6 F5
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum

HEADER = bytes.fromhex("F4F3F2F1")
FOOTER = bytes.fromhex("F8F7F6F5")
BAUD = 256000
TYPE_ENGINEERING = 0x01
TYPE_BASIC = 0x02
MAX_DATA_LEN = 64  # 엔지니어링 모드 문서 예가 35바이트. 이보다 크면 깨진 길이로 봄


class TargetState(IntEnum):
    NONE = 0x00
    MOVING = 0x01
    STATIONARY = 0x02
    MOVING_AND_STATIONARY = 0x03


@dataclass(frozen=True)
class Ld2410Report:
    state: TargetState
    moving_distance_cm: int
    moving_energy: int
    stationary_distance_cm: int
    stationary_energy: int
    detection_distance_cm: int
    moving_gate_energy: tuple[int, ...] = field(default=())     # 엔지니어링 모드에서만
    stationary_gate_energy: tuple[int, ...] = field(default=())

    @property
    def present(self) -> bool:
        return self.state != TargetState.NONE


def parse_data(data: bytes) -> Ld2410Report:
    """프레임 안 데이터(종류 바이트부터 0x00까지) → 보고. 형식이 틀리면 ValueError."""
    if len(data) < 13 or data[0] not in (TYPE_BASIC, TYPE_ENGINEERING) or data[1] != 0xAA or data[-2:] != b"\x55\x00":
        raise ValueError("LD2410C 데이터 형식이 아님")
    state, mdist, menergy, sdist, senergy, ddist = struct.unpack("<BHBHBH", data[2:11])
    moving, stationary = (), ()
    if data[0] == TYPE_ENGINEERING:
        n_move, n_still = data[11], data[12]
        start = 13
        moving = tuple(data[start: start + n_move + 1])
        stationary = tuple(data[start + n_move + 1: start + n_move + 1 + n_still + 1])
        if len(moving) != n_move + 1 or len(stationary) != n_still + 1:
            raise ValueError("LD2410C 엔지니어링 데이터 길이가 맞지 않음")
    return Ld2410Report(TargetState(state), mdist, menergy, sdist, senergy, ddist, moving, stationary)


def build_frame(report: Ld2410Report) -> bytes:
    """mock·테스트용: 보고 → 프레임 (엔지니어링 게이트 에너지가 있으면 엔지니어링 모드)."""
    body = struct.pack(
        "<BHBHBH", report.state, report.moving_distance_cm, report.moving_energy,
        report.stationary_distance_cm, report.stationary_energy, report.detection_distance_cm,
    )
    if report.moving_gate_energy:
        body += bytes([len(report.moving_gate_energy) - 1, len(report.stationary_gate_energy) - 1])
        body += bytes(report.moving_gate_energy) + bytes(report.stationary_gate_energy)
        kind = TYPE_ENGINEERING
    else:
        kind = TYPE_BASIC
    data = bytes([kind, 0xAA]) + body + b"\x55\x00"
    return HEADER + struct.pack("<H", len(data)) + data + FOOTER


class Ld2410StreamParser:
    """직렬 바이트 조각 → 완성된 보고들. 중간부터 읽거나 바이트가 깨져도 다음 헤더에서 다시 맞춘다.

    설정 명령의 ACK 프레임(FD FC FB FA ...)은 헤더가 달라 자연히 건너뛴다.
    """

    def __init__(self):
        self._buf = bytearray()
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[Ld2410Report]:
        self._buf += data
        reports = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                del self._buf[: max(0, len(self._buf) - len(HEADER) + 1)]
                return reports
            if len(self._buf) - start < 6:
                del self._buf[:start]
                return reports
            (length,) = struct.unpack("<H", self._buf[start + 4: start + 6])
            total = 4 + 2 + length + 4
            if length > MAX_DATA_LEN:
                self.bad_frames += 1
                del self._buf[: start + 1]
                continue
            if len(self._buf) - start < total:
                del self._buf[:start]
                return reports
            frame = bytes(self._buf[start: start + total])
            try:
                if frame[-4:] != FOOTER:
                    raise ValueError("끝 바이트가 맞지 않음")
                reports.append(parse_data(frame[6:-4]))
                del self._buf[: start + total]
            except ValueError:
                self.bad_frames += 1
                del self._buf[: start + 1]
