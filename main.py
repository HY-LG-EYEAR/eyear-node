"""EYEAR Node 실행: 카메라 + 마이크(소리 분류, 음성 대화) → 방 상태 메시지와 음성 메시지.

    python main.py --mqtt localhost                 # room1, 카메라 창 + 콘솔
    python main.py --mqtt localhost --no-window     # 창 없이
    python main.py --mqtt localhost --no-camera     # 마이크만 (재실·신원·밝기 = UNKNOWN)
    python main.py --mqtt localhost --no-audio      # 카메라만 (소리 = UNKNOWN, 음성 대화 없음)
    python main.py --mqtt localhost --no-radar      # 레이더 없이 (재실은 카메라만, 위치·움직임 = UNKNOWN)

스레드 구성:
    메인 스레드   카메라 → 신원·자세·활동(RESTING)·밝기 → RoomState → EventBuilder → 방 상태 메시지 (창은 메인 스레드에서만 그릴 수 있음)
    레이더        LD2450·LD2410C 시리얼 → RadarInput → RoomState에서 카메라와 합침 (재실·위치·움직임, 어둠 속 이어가기)
    소리 분류     마이크 → 1초마다 audio.context → RoomState
    음성 대화     마이크 → "하이 엘지" / 질문 대답 → voice/command, voice/reply
                 중앙의 speak 수신 → 스피커 (말하는 동안 마이크를 꺼서 소리 분류도 자기 목소리를 듣지 않음)

원본 영상·음성은 저장하거나 내보내지 않는다 (R-10).
"""

import argparse
import signal
import threading
import time
import traceback

from fusion.events import EventBuilder
from fusion.radar_input import RadarInput
from fusion.room_state import RoomState
from publisher import topics
from publisher.sinks import JsonlSink, MqttSink, MultiSink, Sink, StdoutSink
from schema.enums import Light
from schema.family import load_family


def _thread(name: str, target, *args) -> threading.Thread:
    def run():
        try:
            target(*args)
        except Exception:  # 한 모듈이 죽어도 나머지는 계속 (그 모듈 값은 stale → UNKNOWN)
            print(f"[{name}] 오류로 멈춤:\n{traceback.format_exc()}")

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR Node")
    parser.add_argument("--room", default="room1")
    parser.add_argument("--mqtt", default=None, metavar="HOST[:PORT]", help="없으면 화면 출력만 (질문은 받을 수 없음)")
    parser.add_argument("--jsonl", default=None, metavar="PATH")
    parser.add_argument("--stdout", choices=["summary", "all", "full", "off"], default="summary")
    parser.add_argument("--heartbeat", type=float, default=1.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-audio", action="store_true", help="마이크·스피커를 쓰지 않음 (소리 분류, 음성 대화 모두 끔)")
    parser.add_argument("--no-radar", action="store_true", help="레이더를 쓰지 않음 (포트·설정은 config/radar.yaml)")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-resting", action="store_true", help="휴식 판단(MobileCLIP) 끔 → 누움은 activity UNKNOWN")
    parser.add_argument("--source", default=None, help="카메라 번호 또는 동영상 파일 (기본: config/vision.yaml)")
    args = parser.parse_args()

    family = load_family()
    sinks: list[Sink] = [StdoutSink(args.stdout)] if args.stdout != "off" else []
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    mqtt = None
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        mqtt = MqttSink(host, int(port or 1883))
        sinks.append(mqtt)
    sink = MultiSink(sinks)
    closers = []
    radar = None if args.no_radar else start_radar(closers)
    room = RoomState(family, radar=radar)
    builder = EventBuilder(f"node_{args.room}", args.room, family, args.heartbeat)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # ---------- 마이크: 소리 분류 + 음성 대화 ----------
    if not args.no_audio:
        from audio.classifier import AudioContextWorker, YamnetModel
        from audio.config import load_audio_config
        from audio.mic import MicStream
        from audio.voice_node import build_session

        audio_config = load_audio_config()
        mic = MicStream(audio_config.mic.device)
        yamnet = YamnetModel(audio_config.classifier)
        worker = AudioContextWorker(mic, yamnet.scores, audio_config.classifier, room.set_audio)
        session = build_session(audio_config, mic, sink, args.room)
        if mqtt:
            mqtt.subscribe(topics.speak(args.room), session.on_speak_message)
            print(f"질문 수신: {topics.speak(args.room)}")
        _thread("소리 분류", worker.run, stop)
        _thread("음성 대화", session.run, stop)
        mic.start()
        closers += [mic.close, yamnet.close]
        print("마이크 켜짐: '하이 엘지, 불 켜줘'라고 말해 보세요")

    def publish_state():
        """방 상태를 한 번 계산해 (바뀌었거나 heartbeat 때) 보내고, 화면 표시용으로 돌려준다."""
        snapshot = room.snapshot()
        message = builder.update(snapshot)
        if message:
            sink.publish(topics.state(args.room), message)
        return snapshot

    try:
        if args.no_camera:
            print("카메라 끔 → 재실·신원·밝기는 UNKNOWN (Ctrl+C로 종료)")
            while not stop.is_set():
                publish_state()
                time.sleep(0.1)
        else:
            run_camera(args, family, room, publish_state, stop, closers)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for close in reversed(closers):
            close()
        sink.close()


def start_radar(closers: list) -> RadarInput | None:
    """config/radar.yaml의 포트로 두 레이더를 연다. 연결된 게 없으면 None (레이더 없이 동작)."""
    from pathlib import Path

    from radar.config import load_radar_config
    from radar.ld2410c import Ld2410StreamParser
    from radar.ld2450 import Ld2450StreamParser
    from radar.source import SerialSource

    cfg = load_radar_config()
    radar = RadarInput(cfg)
    p50, p10 = Ld2450StreamParser(), Ld2410StreamParser()

    def feed_50(b: bytes) -> None:
        for targets in p50.feed(b):
            radar.on_ld2450(targets)

    def feed_10(b: bytes) -> None:
        for report in p10.feed(b):
            radar.on_ld2410(report)

    opened = []
    for name, sensor, feed in (("LD2450", cfg.ld2450, feed_50), ("LD2410C", cfg.ld2410c, feed_10)):
        port = sensor.resolve_port()
        if port:
            source = SerialSource(port, feed, sensor.baud).start()
            closers.append(source.close)
            opened.append(name)
        else:
            print(f"{name} 없음 ({sensor.port}) → 이 레이더 값은 UNKNOWN")
    if not opened:
        print("레이더 없음 → 재실은 카메라만, 위치·움직임 = UNKNOWN")
        return None
    where = "설치 위치 있음" if cfg.installation.placed else "설치 위치 미설정 → 위치(position)는 null"
    print(f"레이더 켜짐: {', '.join(opened)} ({where})")
    return radar


def run_camera(args, family, room: RoomState, publish_state, stop: threading.Event, closers: list) -> None:
    import cv2

    from vision.camera import Camera
    from vision.config import load_vision_config
    from vision.demo import draw
    from vision.pipeline import VisionPipeline

    config = load_vision_config()
    if args.source is not None:
        config.camera.source = args.source
    pipeline = VisionPipeline(family, config, resting=not args.no_resting)
    camera = Camera(config.camera)
    closers += [pipeline.close, camera.close]
    if not args.no_window:
        closers.append(cv2.destroyAllWindows)
    print("카메라 켜짐" + ("" if args.no_window else " (창에서 q 또는 ESC로 종료)"))

    fps, last = 0.0, time.monotonic()
    while not stop.is_set():
        frame = camera.read()
        now = time.monotonic()
        if frame is None:
            if camera.is_file:
                break
            room.set_camera(None, Light.UNKNOWN)  # 카메라를 읽을 수 없음 → 지어내지 않고 UNKNOWN
            publish_state()
            time.sleep(0.1)
            continue

        result = pipeline.process(frame, now)
        room.set_camera(result.persons, result.light)
        snapshot = publish_state()

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        if not args.no_window:
            header = (f"{result.light.value} (lum {result.lum:.0f})  occ {snapshot.occupancy.state} "
                      f"{snapshot.occupancy.confidence:.2f}  audio {snapshot.audio.context}  {fps:.1f} fps")
            for k, p in enumerate(snapshot.persons):  # 보내는 값 (카메라 + 레이더 융합 결과)
                pos = "-" if p.position is None else f"({p.position.x:.1f},{p.position.y:.1f})m"
                line = f"SEND #{p.track_id} {p.identity}/{p.identity_source} {p.posture} {p.activity} motion {p.motion} {pos}"
                cv2.putText(frame, line, (10, 55 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (10, 55 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
            draw(frame, result.persons, header, config.pose.min_visibility)
            cv2.imshow("EYEAR node", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break


if __name__ == "__main__":
    main()
