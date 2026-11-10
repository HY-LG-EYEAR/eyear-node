"""1인 방 연속성 테스트."""

from schema.enums import Posture as P
from vision.continuity import SinglePersonContinuity
from vision.tracker import VisionPerson


def cam(track_id, identity="UNKNOWN", box=(0, 0, 100, 200), posture=P.LYING):
    return VisionPerson(track_id=track_id, bbox=box, identity=identity,
                        identity_confidence=0.0 if identity == "UNKNOWN" else 0.8,
                        posture=posture, posture_confidence=0.9)


def ids(out):
    return [(p.track_id, p.identity) for p in out]


def test_camera_track_change_keeps_number_and_identity():
    c = SinglePersonContinuity(hold_seconds=3)
    assert ids(c.update([cam(1, "MEMBER_C")], 0.0)) == [(1, "MEMBER_C")]
    assert ids(c.update([cam(7)], 0.5)) == [(1, "MEMBER_C")]          # 카메라 트랙 1 → 7, 아직 UNKNOWN
    assert ids(c.update([cam(7, "MEMBER_C")], 1.0)) == [(1, "MEMBER_C")]


def test_duplicates_send_only_one_person():
    c = SinglePersonContinuity()
    c.update([cam(1, "MEMBER_C")], 0.0)
    out = c.update([cam(1, "MEMBER_C"), cam(9, box=(0, 0, 300, 300))], 0.5)  # 더 큰 중복이 잡혀도
    assert ids(out) == [(1, "MEMBER_C")] and out[0].bbox == (0, 0, 100, 200)  # 따라가던 트랙 유지


def test_short_loss_is_held_long_loss_clears():
    c = SinglePersonContinuity(hold_seconds=3)
    c.update([cam(1, "MEMBER_C")], 0.0)
    assert ids(c.update([], 2.0)) == [(1, "MEMBER_C")]     # 이불로 잠깐 놓침
    assert ids(c.update([cam(4)], 2.5)) == [(1, "MEMBER_C")]
    assert c.update([], 6.0) == []                          # 3초 넘게 없음 → 나감
    assert ids(c.update([cam(5)], 7.0)) == [(2, "UNKNOWN")]  # 다음 사람은 새 번호, 신원도 새로


def test_different_family_member_replaces_identity():
    c = SinglePersonContinuity()
    c.update([cam(1, "MEMBER_C")], 0.0)
    assert ids(c.update([cam(1, "MEMBER_A")], 0.5)) == [(1, "MEMBER_A")]
