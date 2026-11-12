"""config/radar.yaml 읽기."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RADAR_CONFIG = ROOT / "config" / "radar.yaml"


@dataclass
class SensorConfig:
    port: str | None
    baud: int = 256000
    x_positive: str = "right"   # LD2450만: X > 0 이 레이더 기준 어느 쪽인지 (실측)
    max_range_m: float | None = None  # LD2450만: 이보다 먼 대상은 버림 (소프트웨어 필터)
    max_gate: int | None = None       # LD2410C만: 레이더 내부에 설정해 둔 최대 감지 거리 칸 (기록용, 0.75m 단위)


@dataclass
class RadarConfig:
    ld2450: SensorConfig
    ld2410c: SensorConfig


def load_radar_config(path: Path | str = DEFAULT_RADAR_CONFIG) -> RadarConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RadarConfig(
        ld2450=SensorConfig(**(data.get("ld2450") or {"port": None})),
        ld2410c=SensorConfig(**(data.get("ld2410c") or {"port": None})),
    )
