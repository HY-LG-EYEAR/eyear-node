"""앱이 보낸 사진·녹음 바이트를 메모리에서 바로 푼다 (파일로 쓰지 않음)."""

import io

import numpy as np

SAMPLE_RATE = 16000


def decode_image(data: bytes):
    """jpg/png 등 → BGR 이미지. 읽을 수 없으면 None (HEIC는 앱에서 jpg로 보내야 함)."""
    import cv2

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def decode_audio(data: bytes) -> np.ndarray | None:
    """wav/m4a/aac/mp3 등 → 16kHz 모노 int16. 읽을 수 없으면 None."""
    import av

    chunks: list[np.ndarray] = []
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except (av.error.FFmpegError, ValueError, StopIteration):
        return None
    return np.concatenate(chunks).astype(np.int16) if chunks else None
