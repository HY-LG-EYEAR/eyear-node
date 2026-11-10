"""실제 음성 Node: 마이크·스피커로 "하이 엘지" 명령과 중앙의 질문·대답을 처리한다.

    python -m audio.voice_node --mqtt localhost              # room1
    python -m audio.voice_node --mqtt localhost --room room1 --wav test.wav

publisher/fake_voice.py(가짜 대답)를 실제 마이크·스피커로 바꾼 것. 토픽·메시지 형식은 같다.
"""

import argparse
import signal
import threading

from audio.config import load_audio_config
from audio.mic import MicStream, WavSource
from audio.stt import SpeechRecognizer
from audio.tts import Speaker
from audio.vad import SpeechSegmenter
from audio.voice_session import VoiceSession
from audio.wakeword import TextWakeWord
from publisher import topics
from publisher.sinks import JsonlSink, MqttSink, MultiSink, Sink, StdoutSink


def build_session(config, source, sink: Sink, room_id: str, debug: bool = False) -> VoiceSession:
    print(f"Whisper '{config.stt.model}' 모델 불러오는 중...")
    recognizer = SpeechRecognizer(config.stt)
    return VoiceSession(
        source=source,
        segmenter=SpeechSegmenter(config.vad),
        transcribe=recognizer.transcribe,
        say=Speaker(config.tts).say,
        wakeword=TextWakeWord(config.wakeword.phrases),
        sink=sink,
        node_id=f"node_{room_id}",
        room_id=room_id,
        config=config,
        debug=debug,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 음성 Node")
    parser.add_argument("--room", default="room1")
    parser.add_argument("--mqtt", default=None, metavar="HOST[:PORT]", help="없으면 화면 출력만 (질문은 받을 수 없음)")
    parser.add_argument("--jsonl", default=None, metavar="PATH")
    parser.add_argument("--wav", default=None, help="마이크 대신 WAV 파일 (16kHz 모노 16비트)")
    parser.add_argument("--debug", action="store_true", help="테스트용: 대답 듣는 동안의 소리 크기·말소리 구간·인식 결과 표시")
    args = parser.parse_args()

    config = load_audio_config()
    sinks: list[Sink] = [StdoutSink("summary")]
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    mqtt = None
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        mqtt = MqttSink(host, int(port or 1883))
        sinks.append(mqtt)
    sink = MultiSink(sinks)

    source = WavSource(args.wav) if args.wav else MicStream(config.mic.device)
    session = build_session(config, source, sink, args.room, debug=args.debug)
    if mqtt:
        mqtt.subscribe(topics.speak(args.room), session.on_speak_message)
        print(f"질문 수신: {topics.speak(args.room)}")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    source.start()
    print("듣는 중... '하이 엘지, 불 켜줘'라고 말해 보세요 (Ctrl+C로 종료)")
    try:
        session.run(stop)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        sink.close()


if __name__ == "__main__":
    main()
