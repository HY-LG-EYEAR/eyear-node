"""LD2410C 설정 읽기·바꾸기 (레이더 내부 설정. 공식 문서: 전원을 꺼도 유지됨).

    python -m radar.ld2410c_config --read                 # 현재 설정 보기
    python -m radar.ld2410c_config --max-gate 4           # 최대 감지 거리 4칸 = 3.0m (움직임·정지 둘 다)
    python -m radar.ld2410c_config --max-gate 8           # 공장 기본값 (6.0m)으로 되돌리기

근거: Hi-Link "LD2410C Serial communication protocol" V1.00
- 명령 프레임: FD FC FB FA | 길이(2) | 명령어(2) + 값 | 04 03 02 01, ACK는 명령어 | 0x0100
- 설정 시작 0x00FF(값 0x0001) → 명령 → 설정 끝 0x00FE. 시작 없이 보낸 명령은 무시됨
- 최대 거리·무인 시간 0x0060: 파라미터어 0x0000 움직임 최대 게이트, 0x0001 정지 최대 게이트, 0x0002 무인 시간(초)
  각 (2바이트 파라미터어 + 4바이트 값). 게이트 범위 2~8, 게이트 하나 = 0.75m
- 설정 읽기 0x0061
"""

import argparse
import struct
import time

from radar.config import load_radar_config

CMD_HEADER = bytes.fromhex("FDFCFBFA")
CMD_FOOTER = bytes.fromhex("04030201")
GATE_M = 0.75


def build_command(word: int, value: bytes = b"") -> bytes:
    body = struct.pack("<H", word) + value
    return CMD_HEADER + struct.pack("<H", len(body)) + body + CMD_FOOTER


def find_ack(buf: bytes, word: int) -> bytes | None:
    """buf에서 명령 word에 대한 ACK의 반환값(상태 2바이트부터)을 찾는다."""
    i = 0
    while (i := buf.find(CMD_HEADER, i)) >= 0:
        if len(buf) < i + 6:
            return None
        (length,) = struct.unpack("<H", buf[i + 4:i + 6])
        end = i + 6 + length
        if len(buf) < end + 4:
            return None
        data = buf[i + 6:end]
        if buf[end:end + 4] == CMD_FOOTER and len(data) >= 2 and struct.unpack("<H", data[:2])[0] == (word | 0x0100):
            return data[2:]
        i += 1
    return None


def max_gate_value(move_gate: int, still_gate: int, no_one_seconds: int) -> bytes:
    return struct.pack("<HIHIHI", 0x0000, move_gate, 0x0001, still_gate, 0x0002, no_one_seconds)


def parse_params(ret: bytes) -> dict:
    """0x0061 반환값: 상태(2) AA N 움직임최대 정지최대 움직임감도×(N+1) 정지감도×(N+1) 무인시간(2)."""
    status = struct.unpack("<H", ret[:2])[0]
    if status != 0 or ret[2] != 0xAA:
        raise ValueError("설정 읽기 실패")
    n, move_max, still_max = ret[3], ret[4], ret[5]
    move_sens = list(ret[6:6 + n + 1])
    still_sens = list(ret[6 + n + 1:6 + 2 * (n + 1)])
    (no_one,) = struct.unpack("<H", ret[6 + 2 * (n + 1):8 + 2 * (n + 1)])
    return {"max_gate": n, "move_max_gate": move_max, "still_max_gate": still_max,
            "move_sensitivity": move_sens, "still_sensitivity": still_sens, "no_one_seconds": no_one}


class Ld2410Config:
    def __init__(self, port: str, baud: int):
        import serial

        self.serial = serial.Serial(port, baud, timeout=0.05)

    def send(self, word: int, value: bytes = b"", timeout: float = 1.5) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(build_command(word, value))
        buf, start = b"", time.monotonic()
        while time.monotonic() - start < timeout:
            buf += self.serial.read(256)
            ret = find_ack(buf, word)
            if ret is not None:
                return ret
        raise TimeoutError(f"명령 0x{word:04X}의 응답(ACK)이 없음")

    def session(self, actions):
        """설정 시작 → actions(self) → 설정 끝 (중간에 실패해도 설정 끝은 보냄)."""
        ret = self.send(0x00FF, struct.pack("<H", 0x0001))
        if struct.unpack("<H", ret[:2])[0] != 0:
            raise RuntimeError("설정 시작 실패")
        try:
            return actions(self)
        finally:
            self.send(0x00FE)

    def read_params(self) -> dict:
        return parse_params(self.send(0x0061))

    def close(self) -> None:
        self.serial.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LD2410C 설정")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--max-gate", type=int, choices=range(2, 9), metavar="2~8",
                        help="최대 감지 거리 (게이트 하나 = 0.75m, 4 = 3.0m, 8 = 공장 기본 6.0m)")
    args = parser.parse_args()

    cfg = load_radar_config().ld2410c
    radar = Ld2410Config(cfg.port, cfg.baud)
    try:
        def run(r: Ld2410Config):
            before = r.read_params()
            if args.max_gate:
                ret = r.send(0x0060, max_gate_value(args.max_gate, args.max_gate, before["no_one_seconds"]))
                if struct.unpack("<H", ret[:2])[0] != 0:
                    raise RuntimeError("최대 거리 설정 실패")
            return before, r.read_params()

        before, after = radar.session(run)
    finally:
        radar.close()
    for name, p in (("이전", before), ("현재", after)) if args.max_gate else (("현재", after),):
        print(f"[{name}] 최대 감지 거리: 움직임 {p['move_max_gate']}칸({p['move_max_gate'] * GATE_M:.2f}m), "
              f"정지 {p['still_max_gate']}칸({p['still_max_gate'] * GATE_M:.2f}m), 무인 유지 {p['no_one_seconds']}초")
    print("  거리별 움직임 감도:", after["move_sensitivity"])
    print("  거리별 정지 감도:  ", after["still_sensitivity"], "(값이 클수록 둔감, 100이면 그 거리는 감지 안 함)")


if __name__ == "__main__":
    main()
