"""MobileCLIP (이미지–문장 매칭, 제로샷). 휴식 판단과 사람 확인이 모델 하나를 같이 쓴다.

CLIP은 주어진 문장들 중에서만 고른다. 문장이 모두 "사람이 ~하는 사진"이면 빈 의자도 "의자에 앉은 사람"으로 답하므로
(2026-11-10 실측 0.97), 판단마다 반대쪽 문장(빈 의자, 빈 침대 등)을 함께 넣어야 한다.
화면은 메모리에서만 쓰고 저장하지 않는다 (R-10).
"""

import numpy as np

from vision.config import RestingConfig


def pad_square(rgb: np.ndarray) -> np.ndarray:
    """CLIP 전처리는 가운데를 정사각형으로 잘라내므로, 옆으로 긴 누운 사람이 잘리지 않게 먼저 여백을 채운다."""
    h, w = rgb.shape[:2]
    size = max(h, w)
    out = np.zeros((size, size, 3), rgb.dtype)
    y0, x0 = (size - h) // 2, (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return out


class PromptSet:
    """문장 묶음. probabilities()는 이 문장들 사이의 softmax."""

    def __init__(self, prompts: list[str], features):
        self.prompts = prompts
        self.features = features


class ClipModel:
    def __init__(self, model: str, pretrained: str, device: str = "auto"):
        import open_clip  # torch 포함, 무거워서 실제로 쓸 때만 불러옴
        import torch

        self._torch = torch
        self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        clip, _, self._preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self._model = clip.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(model)
        self._scale = self._model.logit_scale.exp()

    def prompt_set(self, prompts: list[str]) -> PromptSet:
        with self._torch.no_grad():
            text = self._model.encode_text(self._tokenizer(prompts).to(self.device))
        return PromptSet(list(prompts), text / text.norm(dim=-1, keepdim=True))

    def probabilities(self, crop_bgr: np.ndarray, prompts: PromptSet) -> np.ndarray:
        from PIL import Image

        image = Image.fromarray(pad_square(np.ascontiguousarray(crop_bgr[:, :, ::-1])))
        with self._torch.no_grad():
            features = self._model.encode_image(self._preprocess(image).unsqueeze(0).to(self.device))
            features = features / features.norm(dim=-1, keepdim=True)
            probs = (self._scale * features @ prompts.features.T).softmax(dim=-1)[0]
        return probs.cpu().numpy()


def load_clip(config: RestingConfig) -> ClipModel | None:
    """모델을 쓸 수 없으면 None. 그러면 휴식 판단은 UNKNOWN, 사람 확인은 하지 않음 (사람을 놓치지 않게)."""
    if not config.enabled:
        print("CLIP 꺼짐 → 누움은 activity UNKNOWN, 가짜 사람 거르기 안 함")
        return None
    print(f"CLIP 모델 불러오는 중 ({config.model}, 처음에는 다운로드로 1~2분)...")
    try:
        clip = ClipModel(config.model, config.pretrained, config.device)
    except ImportError:
        print("torch/open_clip 없음 → 휴식 판단·가짜 사람 거르기 불가. 설치: pip install -e '.[resting]'")
        return None
    except Exception as e:  # 다운로드 실패 등
        print(f"CLIP 모델을 불러오지 못함 → 휴식 판단·가짜 사람 거르기 불가: {e}")
        return None
    print(f"CLIP 모델 준비됨 ({clip.device})")
    return clip
