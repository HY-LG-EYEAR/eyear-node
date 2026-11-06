"""음성 인식 확인용 데모: 말소리 구간마다 인식한 텍스트를 콘솔에 찍는다.

    python -m audio.listen                     # 마이크로 듣기 (Ctrl+C로 종료)
    python -m audio.listen --wav 파일.wav       # WAV 파일 (16kHz 모노 16비트)
    python -m audio.listen --say "안녕하세요"     # 스피커로 말하기만 하고 끝
    python -m audio.listen --devices           # 마이크 장치 목록
    python -m audio.listen --model large-v3-turbo

이 데모는 확인용이라 들은 말을 모두 화면에 찍는다. 실제 Node는 "하이 엘지" 뒤 발화와 질문의 대답만 내보낸다.
소리는 저장하지 않는다.
"""

import argparse
import queue
import time

from audio.config import load_audio_config
from audio.mic import SAMPLE_RATE, MicStream, WavSource
from audio.stt import SpeechRecognizer
from audio.tts import Speaker
from audio.vad import SpeechSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 음성 인식 데모")
    parser.add_argument("--wav", default=None, help="마이크 대신 WAV 파일")
    parser.add_argument("--say", default=None, help="이 문장을 스피커로 말하고 끝")
    parser.add_argument("--devices", action="store_true", help="오디오 장치 목록")
    parser.add_argument("--model", default=None, help="Whisper 모델 (기본: config/audio.yaml)")
    args = parser.parse_args()

    config = load_audio_config()
    if args.devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.say:
        Speaker(config.tts).say(args.say)
        return
    if args.model:
        config.stt.model = args.model

    print(f"Whisper '{config.stt.model}' 모델 불러오는 중... (처음에는 다운로드 때문에 오래 걸림)")
    recognizer = SpeechRecognizer(config.stt)
    segmenter = SpeechSegmenter(config.vad)
    source = WavSource(args.wav) if args.wav else MicStream(config.mic.device)
    frames = source.subscribe()
    source.start()
    print("듣는 중... 말해 보세요 (Ctrl+C로 종료)")

    try:
        while True:
            try:
                frame = frames.get(timeout=0.5)
            except queue.Empty:
                if isinstance(source, WavSource) and source.finished.is_set():
                    break
                continue
            segment = segmenter.push(frame)
            if segment is None:
                continue
            seconds = len(segment) / SAMPLE_RATE
            started = time.perf_counter()
            result = recognizer.transcribe(segment)
            took = time.perf_counter() - started
            if result is None:
                print(f"[말소리 {seconds:.1f}초] 인식 결과 없음 ({took:.1f}초 걸림)")
            else:
                print(f"[말소리 {seconds:.1f}초] '{result.text}' (신뢰도 {result.confidence:.2f}, {took:.1f}초 걸림)")
    except KeyboardInterrupt:
        pass
    finally:
        source.close()


if __name__ == "__main__":
    main()
