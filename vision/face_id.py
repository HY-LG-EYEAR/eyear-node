"""얼굴 찾기 + 등록된 가족과 비교 (insightface, 사전 학습 모델).

등록 데이터는 data/faces/<이름>.npy 에 얼굴 특징값(정규화된 512차원 벡터)만 저장한다. 사진은 저장하지 않는다.
특징값도 생체 정보이므로 git에 올리지 않는다 (.gitignore의 data/).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from schema.enums import IDENTITY_UNKNOWN
from vision.config import ROOT, FaceConfig

FACE_DIR = ROOT / "data" / "faces"


# ---------- 등록 데이터 ----------

class FaceGallery:
    def __init__(self, embeddings: dict[str, np.ndarray]):
        self.embeddings = embeddings  # 이름 → (N, 512)

    @classmethod
    def load(cls, family: frozenset[str], directory: Path = FACE_DIR) -> "FaceGallery":
        embeddings = {}
        for path in sorted(Path(directory).glob("*.npy")):
            if path.stem not in family:
                raise ValueError(f"family.yaml에 없는 이름의 얼굴 등록 파일: {path.name}")
            embeddings[path.stem] = np.load(path)
        return cls(embeddings)

    @staticmethod
    def save(name: str, embeddings: np.ndarray, directory: Path = FACE_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.npy"
        np.save(path, embeddings.astype(np.float32))
        return path

    def similarities(self, embedding: np.ndarray) -> dict[str, float]:
        """가족별 유사도 = 등록된 사진들 중 가장 비슷한 것의 코사인 유사도."""
        return {name: float(np.max(embs @ embedding)) for name, embs in self.embeddings.items()}


@dataclass
class FaceMatch:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (픽셀)
    identity: str                            # 가족 이름 또는 UNKNOWN
    similarity: float                        # 1등 가족과의 유사도 (판단 못 하면 0)
    confidence: float                        # 0~1, UNKNOWN이면 0
    measured: bool = True                    # False: 얼굴이 너무 작아 비교하지 않음 (신원 판단에 쓰지 않음)


def match_embedding(gallery: FaceGallery, embedding: np.ndarray, config: FaceConfig) -> tuple[str, float, float]:
    """(identity, similarity, confidence). 기준 미달이거나 두 사람이 비슷하게 닮았으면 UNKNOWN.

    등록되지 않은 얼굴도 UNKNOWN으로 보낸다 (GUEST는 쓰지 않음, 2026-11-06 결정).
    """
    scores = sorted(gallery.similarities(embedding).items(), key=lambda kv: kv[1], reverse=True)
    if not scores:
        return IDENTITY_UNKNOWN, 0.0, 0.0
    best_name, best = scores[0]
    second = scores[1][1] if len(scores) > 1 else -1.0
    if best < config.match_threshold or best - second < config.ambiguous_margin:
        return IDENTITY_UNKNOWN, best, 0.0
    # 임시 매핑: 기준값에서 0.5, 기준값+0.3 이상이면 0.99. 단계 8에서 실측으로 보정
    confidence = min(0.99, 0.5 + (best - config.match_threshold) / 0.3 * 0.49)
    return best_name, best, round(confidence, 3)


# ---------- 모델 ----------

class FaceIdentifier:
    def __init__(self, gallery: FaceGallery, config: FaceConfig):
        import warnings

        from insightface.app import FaceAnalysis  # 무거워서 실제로 쓸 때만 불러옴

        # insightface 내부의 scikit-image 사용 방식에 대한 경고. 동작에는 영향 없음
        warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

        self.gallery = gallery
        self.config = config
        self._app = FaceAnalysis(name=config.model, allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=0, det_size=(config.det_size, config.det_size), det_thresh=config.min_det_score)

    def embed(self, frame_bgr: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
        """얼굴마다 (bbox, 정규화된 특징값). 너무 작은 얼굴은 특징값 없이 None."""
        result = []
        for face in self._app.get(frame_bgr):
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            embedding = getattr(face, "normed_embedding", None) if (y2 - y1) >= self.config.min_face_px else None
            result.append(((x1, y1, x2, y2), embedding))
        return result

    def identify(self, frame_bgr: np.ndarray) -> list[FaceMatch]:
        matches = []
        for bbox, embedding in self.embed(frame_bgr):
            if embedding is None:
                matches.append(FaceMatch(bbox, IDENTITY_UNKNOWN, 0.0, 0.0, measured=False))
                continue
            identity, similarity, confidence = match_embedding(self.gallery, embedding, self.config)
            matches.append(FaceMatch(bbox, identity, similarity, confidence))
        return matches
