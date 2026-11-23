"""가족 목소리 등록 (화자 식별용). 녹음은 저장하지 않고 목소리 특징값만 data/voices/<이름>.npy 에 저장한다.

    python -m audio.enroll_voice --name MEMBER_D            # 문장 5개를 읽어 등록 (Enter → 4초 녹음)
    python -m audio.enroll_voice --name MEMBER_D --append   # 기존 등록에 추가 (다른 자리·거리에서 더 읽기)
    python -m audio.enroll_voice --test                     # 말하면 누구로 판단하는지와 사람별 유사도 표시 (기준값 측정)
    python -m audio.enroll_voice --test --wav a.wav b.wav    # WAV 파일로 확인 (16kHz 모노)
    python -m audio.enroll_voice --list
    python -m audio.enroll_voice --delete MEMBER_D

등록은 본인 동의를 받고 한다. 녹음한 소리는 메모리에서 특징값을 뽑은 뒤 바로 버린다 (R-10).
"""

import argparse
import sys
import wave

import numpy as np

from audio.config import load_audio_config
from audio.speaker_id import ROOT, SAMPLE_RATE, OnnxSpeakerEmbedder, SpeakerIdentifier, VoiceGallery
from schema.family import load_family

PROMPTS = [
    "하이 엘지, 불 켜줘",
    "하이 엘지, 에어컨 켜줘",
    "오늘 저녁은 뭐 먹을까요",
    "방 안이 조금 덥네요",
    "내일 아침 일곱 시에 깨워 주세요",
]
RECORD_SECONDS = 4.0
FRAME = 480  # 30ms


def dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64))))) / 32768.0
    return 20 * np.log10(rms) if rms > 0 else -120.0


def trim(audio: np.ndarray, below_peak_db: float = 30.0) -> np.ndarray:
    """앞뒤의 조용한 부분을 잘라 말소리만 남긴다 (30ms 단위, 가장 큰 프레임보다 below_peak_db 이상 작으면 조용함)."""
    n = len(audio) // FRAME
    if n == 0:
        return audio
    levels = np.array([dbfs(audio[i * FRAME:(i + 1) * FRAME]) for i in range(n)])
    loud = np.nonzero(levels >= levels.max() - below_peak_db)[0]
    return audio[loud[0] * FRAME:(loud[-1] + 1) * FRAME]


def record(seconds: float, device) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device)
    sd.wait()
    return audio[:, 0].copy()


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16kHz 모노 16비트 WAV만 가능")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def enroll(name: str, append: bool, embed, gallery: VoiceGallery, device) -> None:
    print(f"\n{name} 목소리 등록. 본인 동의를 받았는지 확인하세요. 녹음은 저장하지 않고 특징값만 남깁니다.")
    print("평소 Node와 이야기할 자리(예: 침대)에서, 평소 목소리로 읽어 주세요.\n")
    samples = []
    for i, prompt in enumerate(PROMPTS, 1):
        while True:
            input(f"[{i}/{len(PROMPTS)}] Enter를 누르고 바로 읽기: \"{prompt}\"")
            audio = trim(record(RECORD_SECONDS, device))
            level, seconds = dbfs(audio), len(audio) / SAMPLE_RATE
            if level < -50 or seconds < 0.8:
                print(f"   소리가 너무 작거나 짧음 ({level:.0f} dBFS, {seconds:.1f}초) → 다시")
                continue
            samples.append(embed(audio))
            print(f"   좋음 ({seconds:.1f}초, {level:.0f} dBFS)")
            del audio  # 녹음은 바로 버림
            break
    count = gallery.save(name, np.stack(samples), append=append)
    print(f"\n저장: data/voices/{name}.npy (특징값 {count}개, 녹음 없음)")
    if len(gallery.samples) > 1:
        cents = gallery.centroids()
        others = {k: float(cents[name] @ v) for k, v in cents.items() if k != name}
        print("다른 사람과의 유사도 (낮을수록 잘 구분됨):", ", ".join(f"{k} {v:.2f}" for k, v in sorted(others.items())))


def test(identifier: SpeakerIdentifier, wavs: list[str], device) -> None:
    def show(audio, label):
        scores = identifier.scores(audio) if len(audio) else {}
        who, conf = identifier.identify(audio)
        ranked = ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
        print(f"{label}: {len(audio) / SAMPLE_RATE:.1f}초 → {who} ({conf:.2f}) | {ranked}")

    if wavs:
        for path in wavs:
            show(trim(read_wav(path)), path)
        return
    print(f"기준값: threshold {identifier.threshold}, margin {identifier.margin}, 최소 {identifier.min_seconds}초. Ctrl+C로 끝")
    while True:
        input("Enter를 누르고 말하기 (3초): ")
        audio = trim(record(3.0, device))
        show(audio, "결과")
        del audio


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 가족 목소리 등록 (화자 식별)")
    parser.add_argument("--name", help="등록할 가족 이름 (config/family.yaml)")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--test", action="store_true", help="누구로 판단하는지 확인")
    parser.add_argument("--wav", nargs="*", default=[], help="--test에 쓸 WAV 파일")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", metavar="NAME")
    args = parser.parse_args()

    config = load_audio_config()
    gallery = VoiceGallery(ROOT / config.speaker.voices_dir)
    if args.list:
        for name, embs in sorted(gallery.samples.items()):
            print(f"{name}: 특징값 {len(embs)}개")
        if not gallery.samples:
            print("등록된 목소리 없음")
        return
    if args.delete:
        print("삭제함" if gallery.delete(args.delete) else "등록되지 않은 이름")
        return

    embed = OnnxSpeakerEmbedder(config.speaker.model_path)
    if args.test:
        if not gallery.samples:
            sys.exit("등록된 목소리 없음")
        s = config.speaker
        test(SpeakerIdentifier(embed, gallery, s.threshold, s.margin, s.min_seconds), args.wav, config.mic.device)
        return
    if not args.name:
        parser.error("--name, --test, --list, --delete 중 하나")
    family = load_family()
    if args.name not in family:
        sys.exit(f"{args.name}: 가족 목록(config/family.yaml)에 없음. 가능: {', '.join(sorted(family))}")
    enroll(args.name, args.append, embed, gallery, config.mic.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료")
