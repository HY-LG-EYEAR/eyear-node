"""카메라 실시간 데모: 신원·자세·밝기를 화면에 표시하고, 상태가 바뀌면 콘솔에 Node 메시지 요약을 찍는다.

    python -m vision.demo                     # 창 + 콘솔
    python -m vision.demo --mqtt localhost    # 실제 Node 메시지를 MQTT로도 전송 (room1)
    python -m vision.demo --no-window         # 창 없이 콘솔만
    python -m vision.demo --source video.mp4  # 카메라 대신 동영상 파일

창에서 q 또는 ESC로 종료. 프레임은 저장하지 않는다.
"""

import argparse
import time

import cv2

from fusion.camera_only import snapshot_from_camera
from fusion.events import EventBuilder
from publisher import topics
from publisher.sinks import JsonlSink, MqttSink, MultiSink, Sink, StdoutSink
from schema.enums import IDENTITY_UNKNOWN
from schema.family import load_family
from vision.camera import Camera
from vision.config import load_vision_config
from vision.face_id import FaceGallery, FaceIdentifier
from vision.light import classify_light
from vision.pose import PoseEstimator
from vision.tracker import CameraTracker, VisionPerson

# 화면에 그릴 뼈대 (MediaPipe 관절 번호)
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
GREEN, YELLOW, WHITE, BLACK = (80, 200, 80), (0, 200, 255), (255, 255, 255), (0, 0, 0)


def _text(img, text, org, color=WHITE, scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


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
        _text(frame, f"{p.posture} {p.posture_confidence:.2f}{sim}", (x1, y1 + 20), color, 0.5)
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
    args = parser.parse_args()

    config = load_vision_config()
    if args.source is not None:
        config.camera.source = args.source
    family = load_family()
    gallery = FaceGallery.load(family)
    if not gallery.embeddings:
        print("등록된 얼굴이 없음 → 모든 사람이 UNKNOWN으로 나옴. 등록: python -m vision.enroll --name MEMBER_A")
    else:
        print("등록된 얼굴:", ", ".join(f"{n}({len(e)}장)" for n, e in gallery.embeddings.items()))

    sinks: list[Sink] = [StdoutSink(args.stdout)] if args.stdout != "off" else []
    if args.jsonl:
        sinks.append(JsonlSink(args.jsonl))
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        sinks.append(MqttSink(host, int(port or 1883)))
    sink = MultiSink(sinks)
    builder = EventBuilder(f"node_{args.room}", args.room, family, args.heartbeat)

    print("모델 불러오는 중...")
    faces_model = FaceIdentifier(gallery, config.face)
    pose_model = PoseEstimator(config.pose)
    tracker = CameraTracker(config.face, config.pose, config.tracking)
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
                message = builder.update(snapshot_from_camera(None, classify_light(None, config.light)[0], family))
                if message:
                    sink.publish(topics.state(args.room), message)
                time.sleep(0.1)
                continue

            light, lum = classify_light(frame, config.light)
            faces = faces_model.identify(frame)
            poses = pose_model.detect(frame, now)
            persons = tracker.update(faces, poses, now)

            message = builder.update(snapshot_from_camera(persons, light, family))
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
        pose_model.close()
        sink.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
