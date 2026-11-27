"""등록 파일이 바뀌었는지 확인 (실행 중인 Node가 다시 켜지 않고 새 등록을 읽게)."""

import time
from pathlib import Path


class FileWatch:
    """파일·폴더(*.npy)의 이름과 수정 시각이 바뀌면 changed()가 한 번 True. 확인은 min_interval초에 한 번만."""

    def __init__(self, *paths: Path | str, min_interval: float = 2.0, clock=time.monotonic):
        self.paths = [Path(p) for p in paths]
        self.min_interval = min_interval
        self.clock = clock
        self._last_check = clock()
        self._signature = self._read()

    def _read(self) -> tuple:
        sig = []
        for path in self.paths:
            if path.is_dir():
                sig += [(str(p), p.stat().st_mtime_ns) for p in sorted(path.glob("*.npy"))]
            elif path.exists():
                sig.append((str(path), path.stat().st_mtime_ns))
        return tuple(sig)

    def changed(self) -> bool:
        now = self.clock()
        if now - self._last_check < self.min_interval:
            return False
        self._last_check = now
        sig = self._read()
        if sig == self._signature:
            return False
        self._signature = sig
        return True

    def reset(self) -> None:
        """반영에 실패했을 때: 다음 확인에서 다시 바뀐 것으로 보게."""
        self._signature = ()
