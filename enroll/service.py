"""등록 처리: 사진·녹음 → 특징값만 저장 (원본은 메모리에서 바로 버림). HTTP와 분리해 테스트할 수 있게 한다.

얼굴: data/faces/<신원>.npy (vision.face_id.FaceGallery와 같은 형식)
목소리: data/voices/<신원>.npy (audio.speaker_id.VoiceGallery와 같은 형식)
가족 목록: config/family.yaml (Node 메시지의 identity 검사에 쓰임)
"""

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from schema.enums import RESERVED_IDENTITIES
from schema.family import DEFAULT_FAMILY_PATH, load_family

ROOT = Path(__file__).resolve().parent.parent
IDENTITY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
MIN_SAMPLES = 3            # 한 번에 이만큼은 쓸 수 있어야 저장 (얼굴 CLI는 5장 권장)
MAX_FILES = 10
VOICE_MIN_SECONDS = 0.8
VOICE_MIN_DBFS = -50.0
VOICE_MAX_SECONDS = 15.0

# 얼굴 특징값 추출: 이미지 → [(bbox, 특징값 또는 None(얼굴이 너무 작음)), ...]
FaceEmbed = Callable[[np.ndarray], list]
# 목소리 특징값 추출: 16kHz int16 → 특징값
VoiceEmbed = Callable[[np.ndarray], np.ndarray]


class EnrollError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Result:
    identity: str
    accepted: int = 0
    rejected: list[dict] = field(default_factory=list)
    saved: bool = False
    total: int = 0

    def as_dict(self) -> dict:
        return {"identity": self.identity, "accepted": self.accepted, "rejected": self.rejected,
                "saved": self.saved, "total": self.total}


def check_identity(identity: str) -> None:
    if not IDENTITY_RE.match(identity or ""):
        raise EnrollError(400, "INVALID_IDENTITY", "신원 ID는 영문 대문자로 시작하고 대문자·숫자·밑줄만 (최대 32자)")
    if identity in RESERVED_IDENTITIES:
        raise EnrollError(400, "RESERVED_IDENTITY", f"{identity}는 예약어라 쓸 수 없음")


def _dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) / 32768.0
    return 20 * np.log10(rms) if rms > 0 else -120.0


class EnrollService:
    def __init__(self, face_embed: Callable[[], FaceEmbed], voice_embed: Callable[[], VoiceEmbed],
                 family_path: Path = DEFAULT_FAMILY_PATH, face_dir: Path = ROOT / "data" / "faces",
                 voice_dir: Path = ROOT / "data" / "voices"):
        """face_embed·voice_embed: 모델을 처음 쓸 때 만드는 함수 (무거워서 필요할 때만 불러옴)."""
        self._face_factory, self._voice_factory = face_embed, voice_embed
        self._face: FaceEmbed | None = None
        self._voice: VoiceEmbed | None = None
        self.family_path, self.face_dir, self.voice_dir = Path(family_path), Path(face_dir), Path(voice_dir)
        self.lock = threading.Lock()   # 모델과 파일은 한 번에 한 요청만

    # ---------- 가족 목록 ----------
    def family(self) -> list[dict]:
        members = sorted(load_family(self.family_path))
        return [{"identity": m, "face_samples": self._count(self.face_dir, m),
                 "voice_samples": self._count(self.voice_dir, m)} for m in members]

    def add_member(self, identity: str) -> bool:
        """가족 목록에 추가. 이미 있으면 False."""
        check_identity(identity)
        with self.lock:
            members = load_family(self.family_path)
            if identity in members:
                return False
            text = self.family_path.read_text(encoding="utf-8")
            header = [line for line in text.splitlines() if line.startswith("#")]
            body = yaml.safe_dump({"members": sorted(members | {identity})}, allow_unicode=True, sort_keys=False)
            tmp = self.family_path.with_suffix(".tmp")
            tmp.write_text("\n".join(header + [body]), encoding="utf-8")
            tmp.replace(self.family_path)   # 반쯤 쓴 파일을 Node가 읽지 않게
            return True

    def delete_member_data(self, identity: str) -> dict:
        """얼굴·목소리 등록만 지운다 (가족 목록에는 남김: 지난 기록의 identity 검사가 깨지지 않게)."""
        check_identity(identity)
        with self.lock:
            out = {}
            for kind, directory in (("face", self.face_dir), ("voice", self.voice_dir)):
                path = directory / f"{identity}.npy"
                out[kind] = path.exists()
                if path.exists():
                    path.unlink()
            return {"identity": identity, "deleted": out}

    # ---------- 등록 ----------
    def enroll_faces(self, identity: str, images: list, append: bool) -> Result:
        """images: decode_image 결과 (읽을 수 없으면 None)."""
        self._require_member(identity)
        self._check_count(images)
        result = Result(identity)
        with self.lock:
            if self._face is None:
                self._face = self._face_factory()
            samples = []
            for i, image in enumerate(images):
                if image is None:
                    result.rejected.append({"index": i, "reason": "UNREADABLE"})
                    continue
                faces = self._face(image)
                if len(faces) == 0:
                    reason = "NO_FACE"
                elif len(faces) > 1:
                    reason = "MULTIPLE_FACES"
                elif faces[0][1] is None:
                    reason = "FACE_TOO_SMALL"
                else:
                    samples.append(np.asarray(faces[0][1], dtype=np.float32))
                    continue
                result.rejected.append({"index": i, "reason": reason})
            return self._save(result, self.face_dir, identity, samples, append)

    def enroll_voices(self, identity: str, clips: list, append: bool) -> Result:
        """clips: decode_audio 결과 (16kHz int16, 읽을 수 없으면 None)."""
        from audio.enroll_voice import trim  # 앞뒤 조용한 부분 자르기 (등록 CLI와 같은 기준)

        self._require_member(identity)
        self._check_count(clips)
        result = Result(identity)
        with self.lock:
            if self._voice is None:
                self._voice = self._voice_factory()
            samples = []
            for i, audio in enumerate(clips):
                if audio is None:
                    result.rejected.append({"index": i, "reason": "UNREADABLE"})
                    continue
                audio = trim(audio)
                seconds = len(audio) / 16000
                if _dbfs(audio) < VOICE_MIN_DBFS:
                    result.rejected.append({"index": i, "reason": "NO_SPEECH"})
                elif seconds < VOICE_MIN_SECONDS:
                    result.rejected.append({"index": i, "reason": "TOO_SHORT"})
                elif seconds > VOICE_MAX_SECONDS:
                    result.rejected.append({"index": i, "reason": "TOO_LONG"})
                else:
                    samples.append(np.asarray(self._voice(audio), dtype=np.float32))
                del audio   # 녹음은 바로 버림
            return self._save(result, self.voice_dir, identity, samples, append)

    # ---------- 내부 ----------
    def _require_member(self, identity: str) -> None:
        check_identity(identity)
        if identity not in load_family(self.family_path):
            raise EnrollError(404, "UNKNOWN_IDENTITY", f"{identity}가 가족 목록에 없음 (먼저 PUT /v1/family/{identity})")

    @staticmethod
    def _check_count(items: list) -> None:
        if not items:
            raise EnrollError(400, "NO_FILES", "파일이 없음 (multipart 필드 files)")
        if len(items) > MAX_FILES:
            raise EnrollError(400, "TOO_MANY_FILES", f"한 번에 최대 {MAX_FILES}개")

    @staticmethod
    def _count(directory: Path, identity: str) -> int:
        path = directory / f"{identity}.npy"
        return int(np.load(path).shape[0]) if path.exists() else 0

    @staticmethod
    def _save(result: Result, directory: Path, identity: str, samples: list, append: bool) -> Result:
        result.accepted = len(samples)
        path = directory / f"{identity}.npy"
        old = np.load(path) if append and path.exists() else None
        if len(samples) < MIN_SAMPLES and old is None:
            result.total = EnrollService._count(directory, identity)
            return result   # 너무 적으면 저장하지 않음 (기존 등록은 그대로)
        if not samples:
            result.total = len(old)
            return result
        stacked = np.stack(samples)
        if old is not None:
            stacked = np.concatenate([old, stacked])
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / f".{identity}.npy.tmp"   # *.npy가 아니라 Node의 등록 읽기에 걸리지 않음
        with open(tmp, "wb") as f:
            np.save(f, stacked.astype(np.float32))
        tmp.replace(path)   # Node가 반쯤 쓴 파일을 읽지 않게
        result.saved, result.total = True, len(stacked)
        return result
