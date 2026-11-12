"""레이더 화면: 레이더가 보는 범위를 위에서 내려다본 그림에 사람(목표)을 점으로 표시한다.

    python -m radar.view --ld2450 /dev/cu.usbserial-0001
    python -m radar.view --ld2450 PORT --ld2410c PORT2
    python -m radar.view --mock

- 아래 가운데가 레이더, 위쪽이 레이더 정면. 원은 1m 간격, 회색 선은 감지 범위(좌우 ±60°, 최대 6m, 공식 문서)
- 화면은 레이더 뒤에서 레이더가 보는 쪽을 내려다본 그림 (화면 오른쪽 = 레이더 기준 오른쪽).
  LD2450의 X 방향은 config/radar.yaml의 x_positive로 맞춘다 (2026-11-12 실측: X > 0 = 레이더 기준 왼쪽)
창에서 q 또는 ESC로 종료.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from radar.config import load_radar_config
from radar.ld2410c import Ld2410StreamParser
from radar.ld2450 import Ld2450StreamParser, within_range
from radar.monitor import RadarState
from radar.source import MockSource, SerialSource

W, H = 700, 640
RADAR = (W // 2, H - 40)
PX_PER_M = (H - 80) / 6.0
COLORS = [(80, 200, 80), (0, 180, 255), (255, 120, 80)]


X_SIGN = 1  # main()에서 설정: 레이더 기준 오른쪽이 화면 오른쪽이 되도록


def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    return int(RADAR[0] + X_SIGN * x_mm / 1000 * PX_PER_M), int(RADAR[1] - y_mm / 1000 * PX_PER_M)


def background() -> np.ndarray:
    img = np.full((H, W, 3), 30, np.uint8)
    for m in range(1, 7):
        cv2.circle(img, RADAR, int(m * PX_PER_M), (70, 70, 70), 1)
        cv2.putText(img, f"{m}m", (RADAR[0] + 4, RADAR[1] - int(m * PX_PER_M) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    for deg in (-60, 60):
        rad = math.radians(deg)
        end = to_px(6000 * math.sin(rad), 6000 * math.cos(rad))
        cv2.line(img, RADAR, end, (90, 90, 90), 1)
    max_range = load_radar_config().ld2450.max_range_m
    if max_range:
        cv2.circle(img, RADAR, int(max_range * PX_PER_M), (0, 120, 160), 2)
        cv2.putText(img, f"limit {max_range:g}m", (RADAR[0] + 6, RADAR[1] - int(max_range * PX_PER_M) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 200), 1)
    cv2.rectangle(img, (RADAR[0] - 18, RADAR[1] - 4), (RADAR[0] + 18, RADAR[1] + 4), (200, 200, 200), -1)
    cv2.putText(img, "RADAR", (RADAR[0] - 24, RADAR[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "radar LEFT", (12, RADAR[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(img, "radar RIGHT", (W - 110, RADAR[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return img


STATE_COLOR = {"NONE": (90, 90, 90), "MOVING": (0, 160, 255), "STATIONARY": (80, 200, 80),
               "MOVING_AND_STATIONARY": (0, 220, 220)}
STATE_TEXT = {"NONE": "NOBODY", "MOVING": "MOVING", "STATIONARY": "STILL", "MOVING_AND_STATIONARY": "MOVING+STILL"}


def draw_ld2410(img: np.ndarray, r) -> None:
    """오른쪽 위: LD2410C 상태 (큰 글씨) + 움직임·정지 에너지 막대 (0~100) + 거리."""
    x0, y0, w = W - 230, 60, 210
    color = STATE_COLOR[r.state.name]
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + 44), color, -1)
    cv2.putText(img, f"LD2410C {STATE_TEXT[r.state.name]}", (x0 + 8, y0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    for k, (name, energy, dist) in enumerate((("move", r.moving_energy, r.moving_distance_cm),
                                              ("still", r.stationary_energy, r.stationary_distance_cm))):
        y = y0 + 64 + k * 34
        cv2.putText(img, f"{name} {energy:3d}  {dist / 100:.2f}m", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.rectangle(img, (x0, y + 6), (x0 + w, y + 16), (70, 70, 70), -1)
        cv2.rectangle(img, (x0, y + 6), (x0 + int(w * energy / 100), y + 16), (0, 160, 255) if k == 0 else (80, 200, 80), -1)
    # LD2410C가 잡은 거리를 레이더 앞 호로 표시 (방향은 모름)
    for dist, c in ((r.moving_distance_cm, (0, 160, 255)), (r.stationary_distance_cm, (80, 200, 80))):
        if r.state.name != "NONE" and dist > 0:
            rad = int(dist / 100 * PX_PER_M)
            cv2.ellipse(img, RADAR, (rad, rad), 0, 210, 330, c, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="레이더 화면 (위에서 본 그림)")
    parser.add_argument("--ld2450", metavar="PORT", help="없으면 config/radar.yaml의 포트")
    parser.add_argument("--ld2410c", metavar="PORT", help="없으면 config/radar.yaml의 포트")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    global X_SIGN
    X_SIGN = -1 if load_radar_config().ld2450.x_positive == "left" else 1

    state = RadarState()
    p50, p10 = Ld2450StreamParser(), Ld2410StreamParser()

    max_range = load_radar_config().ld2450.max_range_m

    def feed_50(b: bytes) -> None:
        for t in p50.feed(b):
            state.on_ld2450(within_range(t, max_range))

    def feed_10(b: bytes) -> None:
        for r in p10.feed(b):
            state.on_ld2410(r)

    sources = [MockSource("ld2450", feed_50), MockSource("ld2410c", feed_10)] if args.mock else []
    if not args.mock:
        cfg = load_radar_config()
        for port, sensor, feed in ((args.ld2450, cfg.ld2450, feed_50), (args.ld2410c, cfg.ld2410c, feed_10)):
            port = port or sensor.resolve_port()
            if port and Path(port).exists():
                sources.append(SerialSource(port, feed, sensor.baud))
            else:
                print(f"포트 없음 (연결 안 됨?): {port or sensor.port}")
    if not sources:
        parser.error("연결된 레이더 포트가 없음 (설정: config/radar.yaml)")
    for s in sources:
        s.start()

    base = background()
    trail: list[tuple[float, tuple[int, int], int]] = []
    try:
        while True:
            img = base.copy()
            now = time.monotonic()
            targets = state.ld2450 or []
            for i, t in enumerate(targets):
                trail.append((now, to_px(t.x_mm, t.y_mm), i))
            trail = [p for p in trail if now - p[0] < 3.0]  # 최근 3초 자취
            for _, pt, i in trail:
                cv2.circle(img, pt, 2, COLORS[i % 3], -1)
            for i, t in enumerate(targets):
                pt = to_px(t.x_mm, t.y_mm)
                cv2.circle(img, pt, 12, COLORS[i], -1)
                label = f"#{i + 1} x={t.x_mm / 1000:+.2f} y={t.y_mm / 1000:.2f}m {t.speed_cms:+d}cm/s"
                cv2.putText(img, label, (pt[0] + 16, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i], 1)
            lines = [f"LD2450: {len(targets)} target(s)   frames {state.frames['ld2450']}"]
            r = state.ld2410
            if r is not None:
                lines.append(f"LD2410C: {r.state.name}  move {r.moving_distance_cm}cm/{r.moving_energy}  "
                             f"still {r.stationary_distance_cm}cm/{r.stationary_energy}   frames {state.frames['ld2410c']}")
            if state.frames["ld2450"] == 0 and args.ld2450:
                lines.append("no LD2450 frames: check wiring (TX<->RX), 5V, port")
            if r is not None:
                draw_ld2410(img, r)
            for k, text in enumerate(lines):
                cv2.putText(img, text, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.imshow("EYEAR radar (top view)", img)
            if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
