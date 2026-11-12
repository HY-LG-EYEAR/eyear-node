"""config/radar.yaml 읽기."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RADAR_CONFIG = ROOT / "config" / "radar.yaml"


# USB 변환기 칩 (USB vendor id, product id). 꽂는 USB 단자에 따라 포트 이름이 바뀌어서 (usbserial-210 → -10) 칩으로 찾는다
ADAPTER_IDS = {"cp2102": (0x10C4, 0xEA60), "ch340": (0x1A86, 0x7523)}


def find_port(adapter: str | None) -> str | None:
    """연결된 시리얼 포트 중 그 칩의 포트. 없거나 둘 이상이면 None."""
    if adapter not in ADAPTER_IDS:
        return None
    from serial.tools import list_ports

    vid, pid = ADAPTER_IDS[adapter]
    found = [p.device for p in list_ports.comports() if (p.vid, p.pid) == (vid, pid)]
    return found[0] if len(found) == 1 else None


@dataclass
class SensorConfig:
    port: str | None
    adapter: str | None = None        # "cp2102" | "ch340": port가 없을 때 이 칩의 포트를 찾아 씀
    baud: int = 256000
    x_positive: str = "right"         # LD2450만: X > 0 이 레이더 기준 어느 쪽인지 (실측)
    max_range_m: float | None = None  # LD2450만: 이보다 먼 대상은 버림 (소프트웨어 필터)
    max_gate: int | None = None       # LD2410C만: 레이더 내부에 설정해 둔 최대 감지 거리 칸 (기록용, 0.75m 단위)

    def resolve_port(self) -> str | None:
        """설정한 포트가 있으면 그것, 없으면 adapter 칩으로 찾은 포트."""
        if self.port and Path(self.port).exists():
            return self.port
        return find_port(self.adapter)


@dataclass
class InstallationConfig:
    radar_x: float | None = None
    radar_y: float | None = None
    facing_deg: float | None = None
    room_width_m: float | None = None
    room_depth_m: float | None = None

    @property
    def placed(self) -> bool:
        return None not in (self.radar_x, self.radar_y, self.facing_deg)

    @property
    def has_room(self) -> bool:
        return None not in (self.room_width_m, self.room_depth_m)


@dataclass
class MotionConfig:
    window_seconds: float = 3.0
    still_below: float = 15
    active_from: float = 60
    active_speed_cms: float = 30
    slight_speed_cms: float = 8
    stale_seconds: float = 2.0


@dataclass
class RadarConfig:
    ld2450: SensorConfig
    ld2410c: SensorConfig
    installation: InstallationConfig = field(default_factory=InstallationConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)


def _build(cls, values):
    values = values or {}
    unknown = values.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"config/radar.yaml {cls.__name__}: 알 수 없는 설정 {sorted(unknown)}")
    return cls(**values)


def load_radar_config(path: Path | str = DEFAULT_RADAR_CONFIG) -> RadarConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RadarConfig(
        ld2450=_build(SensorConfig, data.get("ld2450") or {"port": None}),
        ld2410c=_build(SensorConfig, data.get("ld2410c") or {"port": None}),
        installation=_build(InstallationConfig, data.get("installation")),
        motion=_build(MotionConfig, data.get("motion")),
    )
