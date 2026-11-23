"""녹화 재생 (recorder.replay)과 MQTT 접속 도우미 테스트 (브로커 없이)."""

import json
from datetime import datetime, timezone

from publisher.sinks import new_client, parse_broker
from recorder.replay import Rewriter, load, play, window

SAMPLE = "recorder/samples/2026-11-19_room1_rest_passerby.jsonl"


def test_sample_recording_loads_state_messages_in_time_order():
    records = load(SAMPLE)
    assert len(records) > 200
    assert records[0].at == 0 and 220 < records[-1].at < 230
    assert all(r.topic == "eyear/v0/room1/state" for r in records)
    assert all(a.at <= b.at for a, b in zip(records, records[1:]))


def test_window_restarts_time_at_zero():
    part = window(load(SAMPLE), 76, 116)
    assert part[0].at == 0 and part[-1].at <= 40


def test_rewriter_makes_messages_look_like_a_freshly_started_node():
    records = load(SAMPLE)[:3]
    started = datetime(2026, 11, 26, 14, 0, tzinfo=timezone.utc)
    rw = Rewriter(started)
    out = [rw.rewrite(r, r.at) for r in records]
    assert [m.seq for m in out] == [0, 1, 2]
    assert len({m.boot_id for m in out}) == 1 and str(out[0].boot_id) != records[0].payload["boot_id"]
    assert len({m.event_id for m in out}) == 3
    assert out[0].timestamp == started
    # 내용은 그대로
    assert out[0].occupancy.model_dump(mode="json") == records[0].payload["occupancy"]


class _Collect:
    def __init__(self):
        self.got = []

    def publish(self, topic, message):
        self.got.append((topic, message))

    def close(self):
        pass


def test_play_keeps_gaps_and_speed():
    records = window(load(SAMPLE), 0, 10)
    t = [0.0]
    sink = _Collect()
    play(records, sink, speed=2.0, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s),
         now=lambda: datetime(2026, 11, 26, tzinfo=timezone.utc))
    assert len(sink.got) == len(records)
    assert abs(t[0] - records[-1].at / 2) < 1e-6          # 두 배 빠르게


def test_parse_broker():
    assert parse_broker("192.168.0.5") == ("192.168.0.5", 1883)
    assert parse_broker("localhost:1884") == ("localhost", 1884)


def test_mqtt_credentials_come_from_environment(monkeypatch):
    monkeypatch.delenv("EYEAR_MQTT_USER", raising=False)
    assert new_client()._username is None
    monkeypatch.setenv("EYEAR_MQTT_USER", "eyear")
    monkeypatch.setenv("EYEAR_MQTT_PASSWORD", "secret")
    c = new_client()
    assert (c._username, c._password) == (b"eyear", b"secret")


def test_recording_has_no_raw_media():
    """기록에는 의미 정보만 (R-10)."""
    keys = set()
    for line in open(SAMPLE, encoding="utf-8"):
        keys |= set(json.loads(line)["payload"])
    assert not keys & {"image", "frame", "audio_raw", "wav", "embedding"}
