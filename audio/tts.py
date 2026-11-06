"""음성 출력: macOS `say` (한국어). 중앙이 보낸 텍스트를 Node 스피커로 읽는다."""

import shutil
import subprocess

from audio.config import TtsConfig


class Speaker:
    def __init__(self, config: TtsConfig):
        if shutil.which("say") is None:
            raise RuntimeError("macOS say 명령을 찾을 수 없음")
        self.config = config

    def say(self, text: str) -> None:
        """다 말할 때까지 기다린다."""
        subprocess.run(["say", "-v", self.config.voice, text], check=True)
