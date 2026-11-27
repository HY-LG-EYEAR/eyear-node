"""Node 등록 서버: 앱이 같은 와이파이에서 보낸 얼굴 사진·목소리 녹음으로 가족을 등록한다. 형식은 docs/enroll_api.md.

    python -m enroll.server                      # 0.0.0.0:8770, 연결 코드는 켤 때 화면에 표시
    EYEAR_PAIRING_CODE=123456 python -m enroll.server --port 8770

- 사진·녹음은 메모리에서 특징값만 뽑고 바로 버린다. 파일로 쓰지 않고, 로그에도 남기지 않는다 (R-10).
- 이 서버는 집 와이파이 안에서만 연다. 인터넷(EC2 등)으로 내보내지 않는다.
- 실행 중인 Node(main.py)는 등록 파일이 바뀌면 몇 초 안에 다시 읽는다 (다시 켤 필요 없음).
"""

import argparse
import hmac
import json
import os
import re
import secrets
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from enroll.media import decode_audio, decode_image
from enroll.service import EnrollError, EnrollService

MAX_BODY = 40 * 1024 * 1024
PATH_RE = re.compile(r"^/v1/family(?:/([^/]+))?(?:/(face|voice))?/?$")


def parse_multipart(content_type: str, body: bytes) -> list[bytes]:
    """multipart/form-data에서 파일 부분만 (필드 이름은 보지 않음, 보낸 순서대로)."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise EnrollError(415, "UNSUPPORTED_CONTENT_TYPE", "multipart/form-data로 보내야 함")
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise EnrollError(400, "BAD_MULTIPART", "multipart 형식이 아님")
    return [part.get_payload(decode=True) or b"" for part in msg.iter_parts() if part.get_filename() is not None]


def make_handler(service: EnrollService, pairing_code: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EYEAR-Node-Enroll/0.1"

        def log_message(self, fmt, *args):   # 요청 줄만 (본문·파일 내용은 남기지 않음)
            print(f"[등록] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):   # Expo 웹 미리보기용
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-EYEAR-Pairing")
            self.end_headers()

        def _route(self, method: str) -> None:
            try:
                given = self.headers.get("X-EYEAR-Pairing", "").encode("utf-8", "replace")
                if not hmac.compare_digest(given, pairing_code.encode()):
                    raise EnrollError(401, "BAD_PAIRING_CODE", "연결 코드가 틀림 (Node 화면에 표시된 6자리)")
                url = urlparse(self.path)
                m = PATH_RE.match(url.path)
                if not m:
                    raise EnrollError(404, "NOT_FOUND", "없는 주소")
                identity, kind = m.group(1), m.group(2)
                if method == "GET" and identity is None:
                    return self._send(200, {"members": service.family()})
                if method == "PUT" and identity and kind is None:
                    added = service.add_member(identity)
                    return self._send(201 if added else 200, {"identity": identity, "added": added})
                if method == "DELETE" and identity and kind is None:
                    return self._send(200, service.delete_member_data(identity))
                if method == "POST" and identity and kind:
                    mode = parse_qs(url.query).get("mode", ["replace"])[0]
                    if mode not in ("replace", "append"):
                        raise EnrollError(400, "BAD_MODE", "mode는 replace 또는 append")
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0:
                        raise EnrollError(400, "NO_FILES", "본문이 없음")
                    if length > MAX_BODY:
                        raise EnrollError(413, "TOO_LARGE", f"최대 {MAX_BODY // 1024 // 1024}MB")
                    files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
                    if kind == "face":
                        result = service.enroll_faces(identity, [decode_image(f) for f in files], mode == "append")
                    else:
                        result = service.enroll_voices(identity, [decode_audio(f) for f in files], mode == "append")
                    del files   # 원본은 바로 버림
                    return self._send(200, result.as_dict())
                raise EnrollError(405, "METHOD_NOT_ALLOWED", "이 주소에서 쓸 수 없는 방식")
            except EnrollError as e:
                self._send(e.status, {"error": e.code, "message": e.message})
            except ValueError as e:   # family.yaml 형식 오류 등
                self._send(500, {"error": "NODE_CONFIG", "message": str(e)})

        def do_GET(self):
            self._route("GET")

        def do_PUT(self):
            self._route("PUT")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


def default_service() -> EnrollService:
    def face():
        from vision.config import load_vision_config
        from vision.face_id import FaceGallery, FaceIdentifier

        print("얼굴 모델 불러오는 중...", flush=True)
        return FaceIdentifier(FaceGallery({}), load_vision_config().face).embed

    def voice():
        from audio.config import load_audio_config
        from audio.speaker_id import OnnxSpeakerEmbedder

        print("목소리 모델 불러오는 중...", flush=True)
        return OnnxSpeakerEmbedder(load_audio_config().speaker.model_path)

    return EnrollService(face, voice)


def local_addresses() -> list[str]:
    import socket

    addrs = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))   # 실제로 보내지 않음: 바깥으로 나가는 쪽 주소만 확인
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(addrs)


def main() -> None:
    parser = argparse.ArgumentParser(description="EYEAR Node 등록 서버 (집 와이파이 안에서만)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    code = os.environ.get("EYEAR_PAIRING_CODE") or f"{secrets.randbelow(10 ** 6):06d}"
    if not re.fullmatch(r"\d{6}", code):
        raise SystemExit("EYEAR_PAIRING_CODE는 숫자 6자리")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(default_service(), code))
    where = ", ".join(f"http://{a}:{args.port}" for a in local_addresses()) or f"포트 {args.port}"
    print(f"등록 서버 켜짐: {where}\n연결 코드: {code}  (앱 설정에 입력)\n사진·녹음은 저장하지 않음. Ctrl+C로 종료", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
