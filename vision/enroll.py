"""가족 얼굴 등록. 사진은 저장하지 않고 얼굴 특징값만 data/faces/<이름>.npy 에 저장한다.

    python -m vision.enroll --name MEMBER_A                  # 카메라 앞에서 Space로 촬영 (5장 이상)
    python -m vision.enroll --name MEMBER_A --photos ~/a_사진  # 사진 폴더 (jpg/png, 한 장에 한 사람)
    python -m vision.enroll --name MEMBER_A --append         # 기존 등록에 추가
    python -m vision.enroll --list
    python -m vision.enroll --delete MEMBER_A

iPhone 사진(HEIC)은 먼저 JPG로 바꿀 것:  sips -s format jpeg 사진.heic --out 사진.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from schema.family import load_family
from vision.camera import Camera
from vision.config import load_vision_config
from vision.face_id import FACE_DIR, FaceGallery, FaceIdentifier

MIN_SAMPLES = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def single_embedding(model: FaceIdentifier, image) -> tuple[np.ndarray | None, str]:
    faces = model.embed(image)
    if len(faces) != 1:
        return None, f"얼굴 {len(faces)}개 (한 명만 나와야 함)"
    bbox, embedding = faces[0]
    if embedding is None:
        return None, "얼굴이 너무 작음"
    return embedding, "ok"


def from_camera(model: FaceIdentifier, config) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    print("Space: 촬영 / Enter 또는 q: 끝내기. 정면·좌우·위아래로 조금씩 각도를 바꿔 가며 찍기")
    with Camera(config.camera) as camera:
        while True:
            frame = camera.read()
            if frame is None:
                print("카메라 프레임을 읽을 수 없음")
                break
            faces = model.embed(frame)
            ok = len(faces) == 1 and faces[0][1] is not None
            for bbox, _ in faces:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 80) if ok else (0, 0, 255), 2)
            status = "Space to capture" if ok else "need exactly one clear face"
            cv2.putText(frame, f"{len(samples)} captured | {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("EYEAR enroll", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and ok:
                samples.append(faces[0][1])
                print(f"  {len(samples)}장")
            elif key in (13, ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return samples


def from_photos(model: FaceIdentifier, directory: Path) -> list[np.ndarray]:
    samples = []
    paths = sorted(p for p in directory.expanduser().iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        print(f"사진 없음: {directory} (지원 형식: {', '.join(sorted(IMAGE_SUFFIXES))})")
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  건너뜀 {path.name}: 읽을 수 없음")
            continue
        embedding, reason = single_embedding(model, image)
        if embedding is None:
            print(f"  건너뜀 {path.name}: {reason}")
            continue
        samples.append(embedding)
        print(f"  사용 {path.name}")
    return samples


def report(name: str, samples: np.ndarray, gallery: FaceGallery, threshold: float) -> None:
    """등록 품질: 자기 사진끼리는 비슷해야 하고, 다른 가족과는 달라야 한다."""
    if len(samples) > 1:
        sims = samples @ samples.T
        self_sim = sims[np.triu_indices(len(samples), k=1)].mean()
        print(f"자기 사진끼리 평균 유사도: {self_sim:.2f} (높을수록 좋음)")
    for other, embs in gallery.embeddings.items():
        if other == name:
            continue
        cross = float((samples @ embs.T).max())
        warn = "  ⚠ 기준값 이상 → 두 사람이 헷갈릴 수 있음" if cross >= threshold else ""
        print(f"{other}와의 최대 유사도: {cross:.2f}{warn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 가족 얼굴 등록")
    parser.add_argument("--name", help="family.yaml에 있는 이름 (예: MEMBER_A)")
    parser.add_argument("--photos", type=Path, help="사진 폴더 (없으면 카메라로 촬영)")
    parser.add_argument("--append", action="store_true", help="기존 등록에 추가")
    parser.add_argument("--list", action="store_true", help="등록 현황")
    parser.add_argument("--delete", metavar="NAME", help="등록 삭제")
    args = parser.parse_args()

    family = load_family()
    gallery = FaceGallery.load(family)

    if args.list:
        for name in sorted(family):
            count = len(gallery.embeddings.get(name, []))
            print(f"{name}: {count}장" if count else f"{name}: 미등록")
        return
    if args.delete:
        path = FACE_DIR / f"{args.delete}.npy"
        if path.exists():
            path.unlink()
            print(f"삭제: {path}")
        else:
            print(f"등록 없음: {args.delete}")
        return
    if args.name not in family:
        sys.exit(f"--name은 family.yaml의 이름 중 하나여야 함: {', '.join(sorted(family))}")

    config = load_vision_config()
    print("얼굴 모델 불러오는 중...")
    model = FaceIdentifier(gallery, config.face)
    samples = from_photos(model, args.photos) if args.photos else from_camera(model, config)

    if args.append and args.name in gallery.embeddings:
        samples = list(gallery.embeddings[args.name]) + samples
    if len(samples) < MIN_SAMPLES:
        sys.exit(f"{len(samples)}장뿐이라 저장하지 않음 (최소 {MIN_SAMPLES}장)")

    stacked = np.stack(samples)
    report(args.name, stacked, gallery, config.face.match_threshold)
    path = FaceGallery.save(args.name, stacked)
    print(f"저장: {path} ({len(samples)}장 분량의 특징값, 사진은 저장하지 않음)")


if __name__ == "__main__":
    main()
