"""앱 등록 서버 (enroll/) 테스트. 모델 대신 가짜 특징값 추출기를 쓴다."""

import io
import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from enroll.media import decode_audio, decode_image
from enroll.server import make_handler, parse_multipart
from enroll.service import MIN_SAMPLES, EnrollError, EnrollService
from enroll.watch import FileWatch
from schema.family import load_family

SR = 16000
HEADER = "# 등록된 가족 목록 (테스트)\n# 두 번째 줄\n"


def fake_face(image):
    """이미지 픽셀 값 = 얼굴 수 (99면 너무 작은 얼굴 하나)."""
    n = int(image[0, 0, 0])
    if n == 99:
        return [((0, 0, 1, 1), None)]
    return [((0, 0, 1, 1), np.ones(4, np.float32) / 2) for _ in range(n)]


def fake_voice(audio):
    return np.ones(3, np.float32) / np.sqrt(3)


def img(n):
    return np.full((8, 8, 3), n, np.uint8)


def tone(seconds, amp=8000, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


@pytest.fixture
def service(tmp_path):
    fam = tmp_path / "family.yaml"
    fam.write_text(HEADER + "members:\n  - MEMBER_A\n", encoding="utf-8")
    return EnrollService(lambda: fake_face, lambda: fake_voice, fam, tmp_path / "faces", tmp_path / "voices")


def test_add_member_keeps_header_and_validates(service):
    assert service.add_member("MEMBER_E") is True
    assert service.add_member("MEMBER_E") is False
    assert load_family(service.family_path) == {"MEMBER_A", "MEMBER_E"}
    assert service.family_path.read_text(encoding="utf-8").startswith(HEADER)
    for bad, code in (("UNKNOWN", "RESERVED_IDENTITY"), ("member_e", "INVALID_IDENTITY"), ("", "INVALID_IDENTITY")):
        with pytest.raises(EnrollError) as e:
            service.add_member(bad)
        assert e.value.code == code


def test_faces_reasons_minimum_and_append(service):
    images = [img(1), img(0), img(2), img(99), None, img(1), img(1)]
    r = service.enroll_faces("MEMBER_A", images, append=False)
    assert r.accepted == 3 and r.saved and r.total == 3
    assert r.rejected == [{"index": 1, "reason": "NO_FACE"}, {"index": 2, "reason": "MULTIPLE_FACES"},
                          {"index": 3, "reason": "FACE_TOO_SMALL"}, {"index": 4, "reason": "UNREADABLE"}]
    # 너무 적으면 저장하지 않고 기존 등록 유지
    r = service.enroll_faces("MEMBER_A", [img(1)] * (MIN_SAMPLES - 1), append=False)
    assert not r.saved and r.total == 3
    # append는 적어도 기존에 더함
    r = service.enroll_faces("MEMBER_A", [img(1)], append=True)
    assert r.saved and r.total == 4
    assert np.load(service.face_dir / "MEMBER_A.npy").shape == (4, 4)
    assert not list(service.face_dir.glob("*.tmp"))
    with pytest.raises(EnrollError) as e:
        service.enroll_faces("MEMBER_Z", [img(1)], append=False)
    assert e.value.status == 404


def test_voices_reasons(service):
    clips = [tone(1.5), np.zeros(SR * 2, np.int16), tone(0.3), tone(1.2), tone(2.0), None]
    r = service.enroll_voices("MEMBER_A", clips, append=False)
    assert r.accepted == 3 and r.saved
    assert {x["reason"] for x in r.rejected} == {"NO_SPEECH", "TOO_SHORT", "UNREADABLE"}
    assert service.family()[0] == {"identity": "MEMBER_A", "face_samples": 0, "voice_samples": 3}
    assert service.delete_member_data("MEMBER_A")["deleted"] == {"face": False, "voice": True}


def wav_bytes(audio, rate=SR, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(audio, channels).tobytes())
    return buf.getvalue()


def test_decode_audio_resamples_to_16k_mono():
    out = decode_audio(wav_bytes(tone(1.0, rate=44100), rate=44100, channels=2))
    assert out is not None and out.dtype == np.int16 and abs(len(out) - SR) < 400
    assert decode_audio(b"not audio") is None


def test_decode_image():
    cv2 = pytest.importorskip("cv2")
    ok, png = cv2.imencode(".png", img(1))
    assert decode_image(png.tobytes()).shape == (8, 8, 3)
    assert decode_image(b"not image") is None


def test_file_watch(tmp_path):
    now = [0.0]
    d = tmp_path / "faces"
    d.mkdir()
    w = FileWatch(d, min_interval=2.0, clock=lambda: now[0])
    (d / "A.npy").write_bytes(b"x")
    assert not w.changed()          # 2초 안에는 확인하지 않음
    now[0] = 3
    assert w.changed() and not w.changed()


def multipart(files):
    boundary = "eyearboundary"
    body = b""
    for i, data in enumerate(files):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{i}.bin\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", body


def test_parse_multipart_keeps_order_and_binary():
    ctype, body = multipart([b"\x00\x01\r\n\xff", b"second"])
    assert parse_multipart(ctype, body) == [b"\x00\x01\r\n\xff", b"second"]
    with pytest.raises(EnrollError):
        parse_multipart("application/json", b"{}")


def test_http_end_to_end(service):
    cv2 = pytest.importorskip("cv2")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "123456"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method, path, code="123456", data=None, ctype=None):
        req = urllib.request.Request(base + path, data=data, method=method)
        req.add_header("X-EYEAR-Pairing", code)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert call("GET", "/v1/family", code="000000") == (401, {"error": "BAD_PAIRING_CODE",
                                                                   "message": "연결 코드가 틀림 (Node 화면에 표시된 6자리)"})
        assert call("PUT", "/v1/family/MEMBER_E")[0] == 201
        photos = [cv2.imencode(".jpg", img(1))[1].tobytes()] * 3
        status, body = call("POST", "/v1/family/MEMBER_E/face?mode=replace", data=multipart(photos)[1],
                            ctype=multipart(photos)[0])
        assert status == 200 and body["saved"] and body["total"] == 3
        clips = [wav_bytes(tone(1.5))] * 3
        status, body = call("POST", "/v1/family/MEMBER_E/voice", data=multipart(clips)[1], ctype=multipart(clips)[0])
        assert status == 200 and body["accepted"] == 3
        status, body = call("GET", "/v1/family")
        assert {"identity": "MEMBER_E", "face_samples": 3, "voice_samples": 3} in body["members"]
        assert call("POST", "/v1/family/MEMBER_E/face?mode=x", data=b"x", ctype="multipart/form-data; boundary=b")[0] == 400
        assert call("DELETE", "/v1/family/MEMBER_E")[1]["deleted"] == {"face": True, "voice": True}
    finally:
        server.shutdown()
        server.server_close()
