"""레이더 파서 테스트: 공식 문서의 예시 바이트를 문서에 적힌 값으로 해석하는지, 조각·쓰레기 바이트에 견디는지."""

import pytest

from radar import ld2410c, ld2450
from radar.ld2410c import Ld2410Report, Ld2410StreamParser, TargetState
from radar.ld2450 import Ld2450StreamParser, Ld2450Target
from radar.source import MockSource

# HLK-LD2450 Serial Communication Protocol V1.03, 2.3절 예시
LD2450_DOC_FRAME = bytes.fromhex(
    "AAFF0300" "0E03B186" "10004001" + "00" * 16 + "55CC"
)
# LD2410C Serial communication protocol V1.00, 2.3.2절 예시 (기본 / 엔지니어링 모드)
LD2410_DOC_BASIC = bytes.fromhex("F4 F3 F2 F1 0D 00 02 AA 02 51 00 00 00 00 3B 00 00 55 00 F8 F7 F6 F5")
LD2410_DOC_ENGINEERING = bytes.fromhex(
    "F4 F3 F2 F1 23 00 "
    "01 AA 03 1E 00 3C 00 00 39 00 00 08 08 3C 22 05 03 03 04 03 06 05 00 00 39 10 13 "
    "06 06 08 04 03 05 55 00 "
    "F8 F7 F6 F5"
)


def test_ld2450_doc_example():
    """문서: X -782 mm, Y 1713 mm, 속도 -16 cm/s, 해상도 320 mm. 목표 2·3은 없음."""
    assert ld2450.parse_frame(LD2450_DOC_FRAME) == [Ld2450Target(-782, 1713, -16, 320)]


def test_ld2450_round_trip_and_sign():
    targets = [Ld2450Target(-1200, 3400, 25, 360), Ld2450Target(0, 500, -5, 320), Ld2450Target(800, 0, 0, 320)]
    assert ld2450.parse_frame(ld2450.build_frame(targets)) == targets
    assert ld2450.decode_signed(0x8000 + 10) == 10 and ld2450.decode_signed(10) == -10


def test_ld2410_doc_examples():
    basic = Ld2410StreamParser().feed(LD2410_DOC_BASIC)
    assert basic == [Ld2410Report(TargetState.STATIONARY, 0x51, 0, 0, 0x3B, 0)]
    [eng] = Ld2410StreamParser().feed(LD2410_DOC_ENGINEERING)
    assert (eng.state, eng.moving_distance_cm, eng.moving_energy, eng.stationary_energy) == (
        TargetState.MOVING_AND_STATIONARY, 30, 60, 57)
    assert eng.moving_gate_energy == (0x3C, 0x22, 5, 3, 3, 4, 3, 6, 5)
    assert eng.stationary_gate_energy == (0, 0, 0x39, 0x10, 0x13, 6, 6, 8, 4)


def test_ld2410_round_trip():
    r = Ld2410Report(TargetState.MOVING, 120, 80, 0, 0, 120)
    assert Ld2410StreamParser().feed(ld2410c.build_frame(r)) == [r]


@pytest.mark.parametrize("parser_cls, frames", [
    (Ld2450StreamParser, [LD2450_DOC_FRAME, ld2450.build_frame([])]),
    (Ld2410StreamParser, [LD2410_DOC_BASIC, LD2410_DOC_ENGINEERING]),
])
def test_stream_parser_handles_chunks_and_garbage(parser_cls, frames):
    """중간부터 읽기 시작(앞 바이트 잘림), 쓰레기 바이트, 1바이트씩 들어오는 경우."""
    stream = frames[0][7:] + b"\x00\x12\xAA\xFF\xF4garbage" + b"".join(frames) + frames[0][:10]
    parser = parser_cls()
    results = []
    for i in range(len(stream)):
        results += parser.feed(stream[i:i + 1])
    assert len(results) == len(frames)


def test_ld2410_ignores_command_ack_frames():
    ack = bytes.fromhex("FDFCFBFA" "0400" "FF010000" "04030201")  # 설정 명령 ACK (다른 헤더)
    assert Ld2410StreamParser().feed(ack + LD2410_DOC_BASIC) != []


def test_mock_sources_produce_parseable_frames():
    m2450 = MockSource("ld2450", lambda b: None, still_lost_after=5)
    m2410 = MockSource("ld2410c", lambda b: None)
    assert ld2450.parse_frame(m2450.frame_at(2)) == []                 # 아무도 없음
    assert len(ld2450.parse_frame(m2450.frame_at(7))) == 1             # 걸어오는 중
    assert ld2450.parse_frame(m2450.frame_at(20)) == []                # 가만히 누워 놓침 (가정)
    [r] = Ld2410StreamParser().feed(m2410.frame_at(20))
    assert r.state == TargetState.STATIONARY                           # LD2410C는 정지한 사람도 잡음
