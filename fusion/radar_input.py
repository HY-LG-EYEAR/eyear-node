"""두 레이더의 최근 값을 모아 "재실·위치·움직임 정도" 요약을 만든다 (융합 입력).

- LD2450: 거리 제한(max_range_m) 밖, 방 밖(설치 정보가 있을 때)으로 계산되는 대상은 버림 (벽 너머 오감지).
  여러 대상이 잡히면 직전에 따라가던 위치에 가장 가까운 대상 하나만 씀 (방1은 1인, 중복·가짜 대상 무시)
- 위치: 최근 window_seconds 동안의 중간값 (가만히 있어도 ±1m 흔들림, 2026-11-12 실측).
  설치 정보(레이더 위치·방향)가 없으면 None (지어내지 않음)
- 움직임 정도: LD2410C 움직임 에너지 중간값 + LD2450 속도 (기준값은 config/radar.yaml motion, 실측)
- 레이더 값이 stale_seconds 넘게 안 오면 그 레이더는 없는 것으로 봄 (R-09)
- **한 번도 움직인 적 없는 LD2450 대상은 무시** (2026-11-12 실측: 두 레이더 간섭으로 약 2.1m에 움직임 없는 가짜 대상이
  생겼다 없어졌다 함). 진짜 사람은 걸어 들어오므로 처음 잡힐 때 움직인다. 움직이던 사람을 놓쳤다가 같은 자리 근처에서
  다시 잡으면(가만히 앉음 등) "움직였던 사람"을 이어받는다

좌표 변환 (평면도 기준, 2026-11-12 확정): 레이더 앞 방향 f = (cos θ, sin θ), 레이더 왼쪽 l = (-sin θ, cos θ)
  방 좌표 = 레이더 위치 + Y·f + (레이더 기준 왼쪽 거리)·l,  LD2450 X > 0 = 레이더 기준 왼쪽 (실측, x_positive)
"""

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from radar.config import RadarConfig
from radar.ld2410c import Ld2410Report, TargetState
from radar.ld2450 import Ld2450Target, within_range
from schema.enums import Motion


MOVED_SPEED_CMS = 10      # 이 이상 속도가 한 번이라도 나오면 "움직인 대상"
MOVED_DISTANCE_MM = 500   # 처음 잡힌 위치에서 이만큼 옮겨 가도 "움직인 대상"
SAME_TRACK_MM = 600       # 프레임 사이 이 안이면 같은 대상
TRACK_LOST_SECONDS = 2.0
REMEMBER_SECONDS = 60.0   # 움직였던 사람을 놓친 자리를 기억하는 시간 (다시 잡히면 이어받음)
REMEMBER_MM = 800


@dataclass
class _Track:
    x: int
    y: int
    first_x: int
    first_y: int
    last_t: float
    moved: bool


@dataclass(frozen=True)
class RadarSummary:
    ld2410_available: bool
    ld2410_present: bool          # LD2410C가 사람 있음 (움직임 또는 정지). available일 때만 의미 있음
    ld2450_available: bool
    ld2450_target: bool           # 최근 1초 안에 LD2450 대상이 있었음
    position: tuple[float, float] | None   # 방 좌표 (m). 설치 정보가 없거나 대상이 없으면 None
    motion: Motion
    recent_speed_cms: float = 0.0          # 최근 1초 동안 고른 대상의 가장 빠른 속도 (걸어 나감 판단용)


def to_room(t: Ld2450Target, cfg: RadarConfig) -> tuple[float, float] | None:
    inst = cfg.installation
    if not inst.placed:
        return None
    theta = math.radians(inst.facing_deg)
    forward = t.y_mm / 1000
    left = (t.x_mm if cfg.ld2450.x_positive == "left" else -t.x_mm) / 1000
    x = inst.radar_x + forward * math.cos(theta) - left * math.sin(theta)
    y = inst.radar_y + forward * math.sin(theta) + left * math.cos(theta)
    return x, y


def inside_room(xy: tuple[float, float] | None, cfg: RadarConfig, margin: float = 0.3) -> bool:
    inst = cfg.installation
    if xy is None or not inst.has_room:
        return True  # 판단할 정보가 없으면 버리지 않음
    x, y = xy
    return -margin <= x <= inst.room_width_m + margin and -margin <= y <= inst.room_depth_m + margin


class RadarInput:
    """레이더 스레드에서 on_ld2450/on_ld2410을 부르고, 메시지를 만드는 쪽에서 summary()를 부른다."""

    def __init__(self, config: RadarConfig, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._targets: deque[tuple[float, Ld2450Target | None]] = deque()  # (시각, 고른 대상 또는 없음)
        self._reports: deque[tuple[float, Ld2410Report]] = deque()
        self._last50 = -math.inf
        self._last10 = -math.inf
        self._last_xy: tuple[int, int] | None = None   # 직전에 고른 대상의 레이더 좌표 (mm)
        self._tracks: list[_Track] = []
        self._moved_spots: deque[tuple[float, int, int]] = deque()   # (놓친 시각, x, y)
        self.ignored_static = 0                                     # 무시한 움직임 없는 대상 수 (진단용)

    # ---------- 입력 (레이더 스레드) ----------

    def on_ld2450(self, targets: list[Ld2450Target]) -> None:
        now = self._clock()
        targets = within_range(targets, self.config.ld2450.max_range_m)
        targets = [t for t in targets if inside_room(to_room(t, self.config), self.config)]
        with self._lock:
            targets = self._moved_only(targets, now)
            chosen = self._choose(targets)
            if chosen is not None:
                self._last_xy = (chosen.x_mm, chosen.y_mm)
            self._targets.append((now, chosen))
            self._last50 = now
            self._trim(now)

    def on_ld2410(self, report: Ld2410Report) -> None:
        now = self._clock()
        with self._lock:
            self._reports.append((now, report))
            self._last10 = now
            self._trim(now)

    def _moved_only(self, targets: list[Ld2450Target], now: float) -> list[Ld2450Target]:
        """프레임 사이로 대상을 이어 붙여, 한 번이라도 움직인 대상만 돌려준다."""
        for track in [t for t in self._tracks if now - t.last_t > TRACK_LOST_SECONDS]:
            if track.moved:
                self._moved_spots.append((track.last_t, track.x, track.y))
        self._tracks = [t for t in self._tracks if now - t.last_t <= TRACK_LOST_SECONDS]
        while self._moved_spots and now - self._moved_spots[0][0] > REMEMBER_SECONDS:
            self._moved_spots.popleft()

        kept, used = [], set()
        for target in targets:
            free = [t for t in self._tracks if id(t) not in used]
            track = min(free, key=lambda t: (t.x - target.x_mm) ** 2 + (t.y - target.y_mm) ** 2, default=None)
            if track is None or math.dist((track.x, track.y), (target.x_mm, target.y_mm)) > SAME_TRACK_MM:
                remembered = any(math.dist((x, y), (target.x_mm, target.y_mm)) <= REMEMBER_MM
                                 for _, x, y in self._moved_spots)
                track = _Track(target.x_mm, target.y_mm, target.x_mm, target.y_mm, now, remembered)
                self._tracks.append(track)
            used.add(id(track))
            track.x, track.y, track.last_t = target.x_mm, target.y_mm, now
            if abs(target.speed_cms) >= MOVED_SPEED_CMS or \
                    math.dist((track.first_x, track.first_y), (track.x, track.y)) >= MOVED_DISTANCE_MM:
                track.moved = True
            if track.moved:
                kept.append(target)
            else:
                self.ignored_static += 1
        return kept

    def _choose(self, targets: list[Ld2450Target]) -> Ld2450Target | None:
        if not targets:
            return None
        if self._last_xy is None:
            return min(targets, key=lambda t: t.x_mm ** 2 + t.y_mm ** 2)  # 처음엔 레이더에 가장 가까운 대상
        lx, ly = self._last_xy
        return min(targets, key=lambda t: (t.x_mm - lx) ** 2 + (t.y_mm - ly) ** 2)

    def _trim(self, now: float) -> None:
        window = self.config.motion.window_seconds
        while self._targets and now - self._targets[0][0] > window:
            self._targets.popleft()
        while self._reports and now - self._reports[0][0] > window:
            self._reports.popleft()

    # ---------- 요약 ----------

    def summary(self) -> RadarSummary:
        now = self._clock()
        cfg = self.config.motion
        with self._lock:
            self._trim(now)
            avail10 = now - self._last10 <= cfg.stale_seconds
            avail50 = now - self._last50 <= cfg.stale_seconds
            reports = [r for _, r in self._reports] if avail10 else []
            seen = [(t, x) for t, x in self._targets if x is not None] if avail50 else []

        present = bool(reports) and reports[-1].state != TargetState.NONE
        recent_target = any(now - t <= 1.0 for t, _ in seen)

        position = None
        if seen and recent_target:
            xs = [x.x_mm for _, x in seen]
            ys = [x.y_mm for _, x in seen]
            med = Ld2450Target(int(statistics.median(xs)), int(statistics.median(ys)), 0, 0)
            xy = to_room(med, self.config)
            position = None if xy is None else (round(xy[0], 2), round(xy[1], 2))

        speed = statistics.median(abs(x.speed_cms) for _, x in seen) if seen else None
        motion = self._motion(reports, present, speed, avail10, avail50 and recent_target)
        recent_speed = max((abs(x.speed_cms) for t, x in seen if now - t <= 1.0), default=0.0)
        return RadarSummary(avail10, present, avail50, avail50 and recent_target, position, motion, recent_speed)

    def _motion(self, reports, present, speed, avail10, has_target) -> Motion:
        cfg = self.config.motion
        if speed is not None and speed >= cfg.active_speed_cms:
            return Motion.ACTIVE
        if avail10 and present:
            energy = statistics.median(r.moving_energy for r in reports)
            if energy >= cfg.active_from:
                return Motion.ACTIVE
            return Motion.SLIGHT if energy >= cfg.still_below else Motion.STILL
        if has_target and speed is not None:  # LD2410C가 없을 때: LD2450 속도만으로
            return Motion.SLIGHT if speed >= cfg.slight_speed_cms else Motion.STILL
        return Motion.UNKNOWN
