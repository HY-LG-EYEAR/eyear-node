"""카메라 실시간 데모: 신원·자세·밝기를 화면에 표시하고, 상태가 바뀌면 콘솔에 Node 메시지 요약을 찍는다.

    python -m vision.demo                     # 창 + 콘솔
    python -m vision.demo --mqtt localhost    # 실제 Node 메시지를 MQTT로도 전송 (room1)
    python -m vision.demo --no-window         # 창 없이 콘솔만
    python -m vision.demo --source video.mp4  # 카메라 대신 동영상 파일

창에서 q 또는 ESC로 종료. 프레임은 저장하지 않는다.
카메라만 따로 확인하는 용도. 카메라·마이크·음성 대화를 합친 실제 Node는 main.py.
"""

import argparse
import time

import cv2

from fusion.events import EventBuilder
from fusion.room_state import build_snapshot
from publisher import topics
from publisher.sinks import JsonlSink, MqttSink, MultiSink, Sink, StdoutSink
from schema.enums import IDENTITY_UNKNOWN, Light
from schema.family import load_family
from vision.camera import Camera
from vision.config import load_vision_config
from vision.pipeline import VisionPipeline
from vision.tracker import VisionPerson

# 화면에 그릴 뼈대 (MediaPipe 관절 번호)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
GREEN, YELLOW, WHITE, BLACK = (80, 200, 80), (0, 200, 255), (255, 255, 255), (0, 0, 0)


def _text(img, text, org, color=WHITE, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _v(value) -> str:
    """enum이면 값만 ("Posture.LYING" → "LYING")."""
    return getattr(value, "value", value)


def draw(frame, persons: list[VisionPerson], header: str, min_vis: float) -> None:
    for p in persons:
        known = p.identity != IDENTITY_UNKNOWN
        color = GREEN if known else YELLOW
        if p.pose is not None:
            for a, b in SKELETON:
                if p.pose.visibility[a] >= min_vis and p.pose.visibility[b] >= min_vis:
                    cv2.line(frame, tuple(map(int, p.pose.points[a])), tuple(map(int, p.pose.points[b])), color, 2)
        if p.face is not None:
            x1, y1, x2, y2 = map(int, p.face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        x1, y1 = int(p.bbox[0]), max(int(p.bbox[1]) - 30, 20)
        _text(frame, f"#{p.track_id} {p.identity} {p.identity_confidence:.2f}", (x1, y1), color)
        sim = f"  sim {p.face.similarity:.2f}" if p.face is not None and p.face.measured else ""
        _text(frame, f"{_v(p.posture)} {p.posture_confidence:.2f}{sim}", (x1, y1 + 20), color, 0.5)
        rest = "" if p.rest_probability is None else f"  rest p {p.rest_probability:.2f}"
        held = "" if p.rest_agree_seconds is None or p.activity == "RESTING" else f" ({p.rest_agree_seconds:.1f}s)"
        _text(frame, f"{_v(p.activity)} {p.activity_confidence:.2f}{rest}{held}", (x1, y1 + 40), color, 0.5)
        if p.person_probability is not None:  # 얼굴이 안 잡힌 사람: CLIP "사람" 확률 (빈 의자 등 거르기)
            _text(frame, f"person p {p.person_probability:.2f}", (x1, y1 + 60), color, 0.5)
    _text(frame, header, (10, 25))


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR 카메라 데모")
    parser.add_argument("--source", default=None, help="카메라 번호 또는 동영상 파일 (기본: config/vision.yaml)")
    parser.add_argument("--room", default="room1")
    parser.add_argument("--mqtt", default=None, metavar="HOST[:PORT]")
    parser.add_argument("--jsonl", default=None, metavar="PATH")
    parser.add_argument("--stdout", choices=["summary", "all", "full", "off"], default="summary")
    parser.add_argument("--heartbeat", type=float, default=1.0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-resting", action="store_true", help="휴식 판단(MobileCLIP) 끔")
    args = parser.parse_args()

    config = load_vision_config()
    if args.source is not None:
        config.camera.source = args.source
    family = load_family()

    sinks: list[Sink] = [StdoutSink(args.stdout)] if args.stdout != "off" else []
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        sinks.append(MqttSink(host, int(port or 1883)))
    sink = MultiSink(sinks)
    builder = EventBuilder(f"node_{args.room}", args.room, family, args.heartbeat)

    pipeline = VisionPipeline(family, config, resting=not args.no_resting)
    camera = Camera(config.camera)
    print("시작 (창에서 q 또는 ESC로 종료)")

    fps, last = 0.0, time.monotonic()
    try:
        while True:
            frame = camera.read()
            now = time.monotonic()
            if frame is None:
                if camera.is_file:
                    break
                # 카메라를 읽을 수 없음 → 지어내지 않고 UNKNOWN
                message = builder.update(build_snapshot(None, Light.UNKNOWN, None, family))
                if message:
                    sink.publish(topics.state(args.room), message)
                time.sleep(0.1)
                continue

            result = pipeline.process(frame, now)
            persons, light, lum = result.persons, result.light, result.lum

            # 카메라 단독 데모: 소리는 UNKNOWN (마이크까지 합친 실행은 main.py)
            message = builder.update(build_snapshot(persons, light, None, family))
            if message:
                sink.publish(topics.state(args.room), message)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            if not args.no_window:
                draw(frame, persons, f"{light.value} (lum {lum:.0f})  persons {len(persons)}  {fps:.1f} fps",
                     config.pose.min_visibility)
                cv2.imshow("EYEAR camera", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        pipeline.close()
        sink.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
