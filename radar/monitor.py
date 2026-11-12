"""레이더 실시간 확인: 받은 값을 0.5초마다 찍고, 원하면 모든 프레임을 JSONL로 기록한다.

    python -m radar.monitor --list                                   # 연결된 시리얼 포트 목록
    python -m radar.monitor --ld2450 /dev/cu.usbserial-XXXX          # LD2450만
    python -m radar.monitor --ld2410c /dev/cu.usbserial-YYYY         # LD2410C만
    python -m radar.monitor --ld2450 PORT1 --ld2410c PORT2 --log out/radar.jsonl
    python -m radar.monitor --mock                                    # 실물 없이 (가짜 바이트)

레이더 값(숫자)만 다루며 영상·음성은 없다.
"""

import argparse
import json
import threading
import time
from pathlib import Path

from radar.config import load_radar_config
from radar.ld2410c import Ld2410Report, Ld2410StreamParser
from radar.ld2450 import Ld2450StreamParser, Ld2450Target, within_range
from radar.source import MockSource, SerialSource, list_ports


class RadarState:
    """두 레이더의 최신 값 (스레드 안전). 프레임 수로 연결 상태도 확인."""

    def __init__(self, log_path: str | None = None):
        self._lock = threading.Lock()
        self.start = time.monotonic()
        self.ld2450: list[Ld2450Target] | None = None
        self.ld2410: Ld2410Report | None = None
        self.frames = {"ld2450": 0, "ld2410c": 0}
        self._log = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "w")

    def _write(self, rec: dict) -> None:
        if self._log:
            self._log.write(json.dumps(rec) + "\n")

    def on_ld2450(self, targets: list[Ld2450Target]) -> None:
        with self._lock:
            self.ld2450 = targets
            self.frames["ld2450"] += 1
            self._write({"t": round(time.monotonic() - self.start, 2), "sensor": "ld2450",
                         "targets": [[t.x_mm, t.y_mm, t.speed_cms, t.resolution_mm] for t in targets]})

    def on_ld2410(self, report: Ld2410Report) -> None:
        with self._lock:
            self.ld2410 = report
            self.frames["ld2410c"] += 1
            self._write({"t": round(time.monotonic() - self.start, 2), "sensor": "ld2410c",
                         "state": report.state.name, "move_cm": report.moving_distance_cm,
                         "move_e": report.moving_energy, "still_cm": report.stationary_distance_cm,
                         "still_e": report.stationary_energy, "detect_cm": report.detection_distance_cm})

    def line(self) -> str:
        with self._lock:
            el = time.monotonic() - self.start
            parts = [f"{el:6.1f}s"]
            if self.ld2450 is not None:
                ts = " ".join(f"({t.x_mm / 1000:+.2f},{t.y_mm / 1000:.2f})m {t.speed_cms:+d}cm/s" for t in self.ld2450)
                parts.append(f"LD2450 {len(self.ld2450)}명 {ts or '-'}")
            if self.ld2410 is not None:
                r = self.ld2410
                parts.append(f"LD2410C {r.state.name} 움직임 {r.moving_distance_cm}cm/{r.moving_energy} "
                             f"정지 {r.stationary_distance_cm}cm/{r.stationary_energy}")
            parts.append(f"[프레임 {self.frames['ld2450']}/{self.frames['ld2410c']}]")
            return " | ".join(parts)

    def close(self) -> None:
        if self._log:
            self._log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="레이더 실시간 확인")
    parser.add_argument("--list", action="store_true", help="시리얼 포트 목록")
    parser.add_argument("--ld2450", metavar="PORT", help="없으면 config/radar.yaml의 포트")
    parser.add_argument("--ld2410c", metavar="PORT", help="없으면 config/radar.yaml의 포트")
    parser.add_argument("--mock", action="store_true", help="실물 대신 가짜 바이트 (두 레이더 모두)")
    parser.add_argument("--log", metavar="PATH", help="모든 프레임을 JSONL로 기록")
    args = parser.parse_args()

    if args.list:
        ports = list_ports()
        print("\n".join(ports) if ports else "시리얼 포트 없음 (USB 어댑터가 연결됐는지, 드라이버가 필요한지 확인)")
        return

    state = RadarState(args.log)
    p2450, p2410 = Ld2450StreamParser(), Ld2410StreamParser()

    max_range = None if args.mock else load_radar_config().ld2450.max_range_m

    def feed_2450(data: bytes) -> None:
        for targets in p2450.feed(data):
            state.on_ld2450(within_range(targets, max_range))

    def feed_2410(data: bytes) -> None:
        for report in p2410.feed(data):
            state.on_ld2410(report)

    sources = []
    if args.mock:
        sources = [MockSource("ld2450", feed_2450), MockSource("ld2410c", feed_2410)]
    if not args.mock:
        cfg = load_radar_config()
        for port, sensor, feed in ((args.ld2450, cfg.ld2450, feed_2450), (args.ld2410c, cfg.ld2410c, feed_2410)):
            port = port or sensor.port
            if port and Path(port).exists():
                sources.append(SerialSource(port, feed, sensor.baud))
            elif port:
                print(f"포트 없음 (연결 안 됨?): {port}")
    if not sources:
        parser.error("연결된 레이더 포트가 없음 (포트 목록: --list, 설정: config/radar.yaml)")

    for s in sources:
        s.start()
    print("0.5초마다 표시 (Ctrl+C로 종료). 프레임 수가 늘지 않으면 배선(TX↔RX 엇갈림)·전원(5V)·포트를 확인")
    try:
        while True:
            time.sleep(0.5)
            print(state.line(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.close()
        state.close()
        bad = p2450.bad_frames + p2410.bad_frames
        if bad:
            print(f"깨진 프레임 {bad}개 (LD2450 {p2450.bad_frames}, LD2410C {p2410.bad_frames})")


if __name__ == "__main__":
    main()
