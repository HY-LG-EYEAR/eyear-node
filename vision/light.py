"""화면 평균 밝기로 방 밝기 판단."""

import cv2
import numpy as np

from schema.enums import Light
from vision.config import LightConfig


def mean_luminance(frame_bgr: np.ndarray) -> float:
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def classify_light(frame_bgr: np.ndarray | None, config: LightConfig) -> tuple[Light, float | None]:
    """(밝기 enum, 평균 밝기 0~255). 프레임이 없으면 UNKNOWN."""
    if frame_bgr is None:
        return Light.UNKNOWN, None
    lum = mean_luminance(frame_bgr)
    if lum < config.dark_below:
        return Light.DARK, lum
    if lum > config.bright_above:
        return Light.BRIGHT, lum
    return Light.DIM, lum
