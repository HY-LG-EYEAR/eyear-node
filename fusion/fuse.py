"""카메라 + 레이더 융합 (방1, 1인 방). 판단 규칙은 2026-11-12 사용자 확정.

재실 (occupancy):
    카메라와 LD2410C 둘 다 사람        → OCCUPIED 0.95
    LD2410C만 (이불·어둠·카메라 밖)   → OCCUPIED 0.8 (LD2450도 보이면 0.85)   ← 레이더 우선
    카메라만 (LD2410C는 없음)          → OCCUPIED 0.8
    LD2410C 없음 + 카메라도 못 봄      → EMPTY 0.9   ← LD2450 혼자만 보이는 대상은 무시
                                          (2026-11-12 실측: 두 레이더 간섭으로 LD2450에 가끔 가짜 대상)
    LD2410C 연결 안 됨                 → LD2450 0.6 / 밝은데 카메라·LD2450 모두 없음 EMPTY 0.7 / 그 밖 UNKNOWN

사람 (1인 방):
    - 번호는 이 클래스가 정함. 방이 EMPTY가 됐다가 다시 사람이 생기면 새 번호 (나갔다 들어온 것)
    - 카메라가 보면: 신원·자세·활동은 카메라, 위치·움직임 정도는 레이더
    - 카메라가 못 보는데 레이더는 사람 있음 (불 꺼짐, 이불, 카메라 밖):
        이번 재실 중 카메라로 본 적 있으면 이어감 — 신원은 TRACK_CARRIED(신뢰도 낮춤), 자세 UNKNOWN(지어내지 않음),
        RESTING은 움직임이 ACTIVE로 active_exit_seconds 이상 이어지기 전까지 유지 (일어났을 수 있음 → UNKNOWN)
        본 적 없으면 (어두운 방에 들어옴): LD2450 대상이 있을 때만 신원 UNKNOWN인 사람, 없으면 persons [] (위치 모름)
    - 한 번 사람을 보낸 뒤에는 방이 EMPTY가 될 때까지 계속 같은 사람으로 보낸다 (위치·신원은 모르면 null·UNKNOWN).
      (2026-11-12 실측: 들어올 때 레이더가 먼저 잡고 잠깐 놓친 뒤 카메라가 잡아 PERSON_ENTERED → LEFT → ENTERED가 3초 안에 나감)
"""

from dataclasses import dataclass

from fusion.radar_input import RadarSummary
from schema.enums import IDENTITY_UNKNOWN, Activity, Light, Motion
from vision.tracker import VisionPerson

CARRIED_IDENTITY_FACTOR = 0.8
CARRIED_ACTIVITY_FACTOR = 0.8
ACTIVE_EXIT_SECONDS = 3.0


@dataclass
class _Carried:
    identity: str = IDENTITY_UNKNOWN
    identity_confidence: float = 0.0
    activity: Activity = Activity.UNKNOWN
    activity_confidence: float = 0.0


class SingleRoomFusion:
    def __init__(self):
        self._id = 0
        self._present = False
        self._carried: _Carried | None = None
        self._active_since: float | None = None
        self._emitted = False  # 이번 재실 중 사람을 한 번이라도 보냄

    def fuse(self, camera: list[VisionPerson] | None, light: Light, radar: RadarSummary, now: float) -> dict:
        """RoomSnapshot의 occupancy·persons 부분 (dict)."""
        cam_seen = bool(camera)
        r10 = radar.ld2410_present if radar.ld2410_available else None
        r50 = radar.ld2450_target if radar.ld2450_available else None

        occupancy = self._occupancy(cam_seen, camera is not None, light, r10, r50)
        if occupancy["state"] != "OCCUPIED":
            self._present, self._carried, self._active_since, self._emitted = False, None, None, False
            return {"occupancy": occupancy, "persons": []}

        if not self._present:  # 새로 들어온 사람
            self._present, self._carried = True, None
            self._id += 1
        self._active_since = (self._active_since or now) if radar.motion == Motion.ACTIVE else None
        position = None if radar.position is None else {"x": radar.position[0], "y": radar.position[1]}

        if cam_seen:
            p = max(camera, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
            if self._carried is None:
                self._carried = _Carried()
            if p.identity != IDENTITY_UNKNOWN:
                self._carried.identity, self._carried.identity_confidence = p.identity, p.identity_confidence
            self._carried.activity, self._carried.activity_confidence = Activity(p.activity), p.activity_confidence
            person = {
                "identity": p.identity,
                "identity_confidence": p.identity_confidence,
                "identity_source": "UNKNOWN" if p.identity == IDENTITY_UNKNOWN else "FACE",
                "posture": p.posture,
                "posture_confidence": p.posture_confidence,
                "activity": p.activity,
                "activity_confidence": p.activity_confidence,
            }
        elif self._carried is not None:
            c = self._carried
            if c.activity == Activity.RESTING and self._active_since is not None \
                    and now - self._active_since >= ACTIVE_EXIT_SECONDS:
                c.activity, c.activity_confidence = Activity.UNKNOWN, 0.0  # 크게 움직임 → 일어났을 수 있음
            keep_rest = c.activity == Activity.RESTING
            known = c.identity != IDENTITY_UNKNOWN
            person = {
                "identity": c.identity,
                "identity_confidence": round(c.identity_confidence * CARRIED_IDENTITY_FACTOR, 3) if known else 0.0,
                "identity_source": "TRACK_CARRIED" if known else "UNKNOWN",
                "posture": "UNKNOWN",
                "posture_confidence": 0.0,
                "activity": "RESTING" if keep_rest else "UNKNOWN",
                "activity_confidence": round(c.activity_confidence * CARRIED_ACTIVITY_FACTOR, 3) if keep_rest else 0.0,
            }
        elif radar.ld2450_target or self._emitted:
            person = {
                "identity": IDENTITY_UNKNOWN, "identity_confidence": 0.0, "identity_source": "UNKNOWN",
                "posture": "UNKNOWN", "posture_confidence": 0.0,
                "activity": "UNKNOWN", "activity_confidence": 0.0,
            }
        else:
            return {"occupancy": occupancy, "persons": []}  # 사람은 있는데 위치·신원 모름

        self._emitted = True
        person |= {"track_id": self._id, "motion": radar.motion, "position": position}
        return {"occupancy": occupancy, "persons": [person]}

    @staticmethod
    def _occupancy(cam_seen: bool, cam_available: bool, light: Light, r10: bool | None, r50: bool | None) -> dict:
        def occ(state: str, confidence: float) -> dict:
            return {"state": state, "confidence": confidence}

        if r10 is not None:
            if cam_seen and r10:
                return occ("OCCUPIED", 0.95)
            if r10:
                return occ("OCCUPIED", 0.85 if r50 else 0.8)
            if cam_seen:
                return occ("OCCUPIED", 0.8)
            return occ("EMPTY", 0.9)
        # LD2410C를 쓸 수 없음
        if cam_seen:
            return occ("OCCUPIED", 0.9 if r50 else 0.8)
        if r50:
            return occ("OCCUPIED", 0.6)
        if cam_available and light == Light.BRIGHT and r50 is False:
            return occ("EMPTY", 0.7)
        return occ("UNKNOWN", 0.0)
