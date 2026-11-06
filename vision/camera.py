"""카메라 입력. 프레임은 메모리에서만 처리하고 저장하지 않는다 (R-10).

source에 동영상 파일 경로를 주면 카메라 대신 파일을 읽는다 (테스트·재생용).
"""

import cv2
import numpy as np

from vision.config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig):
        source = config.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.is_file = isinstance(source, str)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없음: {source}. macOS 설정 > 개인정보 보호 및 보안 > 카메라에서 "
                "터미널(또는 VS Code)의 카메라 접근을 허용했는지 확인"
            )
        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self) -> np.ndarray | None:
        """BGR 프레임. 읽기 실패(카메라 분리, 파일 끝)면 None."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
