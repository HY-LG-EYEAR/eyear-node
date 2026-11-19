"""카메라 + 레이더 융합 (방1, 1인 방). 판단 규칙은 2026-11-12 사용자 확정.

재실 (occupancy) — 시작·유지·끝을 나눈다 (2026-11-12 사용자 확정):
    LD2410C는 레이더 뒤쪽(벽 너머 거실 등) 사람도 잡고 방향을 모름. LD2450은 앞만 봄 (실측: 뒤쪽 0%, 앞의 누운 사람 100%)
    시작: 카메라가 보거나, LD2450이 앞에서 보고 LD2410C도 사람 있음 (LD2410C 혼자로는 시작 안 함 → 뒤쪽 무시)
          LD2450 혼자만 보이는 대상(LD2410C 없음)도 무시 (두 레이더 간섭으로 가끔 가짜 대상)
    유지: 카메라·LD2450·LD2410C 중 하나라도 사람 있음 (어둠 속 수면, 이불 → LD2410C로 유지)
          LD2410C가 "없음"이어도 이미 있던 사람을 LD2450이 방 안에서 보면 유지 (2026-11-19 시연 공간 실측: 3.2m에
          똑바로 누운 사람을 LD2410C는 87~94% 동안 "없음", LD2450은 88~100% 잡음). LD2450이 가만히 있던 사람을
          LD2450_HOLD_SECONDS 안으로 잠깐 놓친 것도 유지 (실측 최대 0.9초). LD2450 혼자로는 시작하지 않음 (가짜 대상)
    끝:   ① LD2410C까지 모두 없음, 또는
          ② 카메라·LD2450이 움직이던(사라지기 전 3초 안에 ACTIVE·WALKING·걷는 속도) 사람을 둘 다 놓치고
             EXIT_SECONDS 동안 다시 못 봄 = 걸어 나감
             (LD2410C가 아직 "있음"이어도: 레이더 뒤쪽 사람일 수 있음. 2026-11-12 실측: 카메라에만 보이던 사람이
              레이더 뒤로 걸어간 뒤 LD2410C 뒤쪽 감지로 계속 "있음"이었음 → 카메라로 본 마지막 순간도 같이 봄)
             가만히·조금 움직이던 중 사라지면(누움, 이불, 소등) 나감이 아님 → LD2410C로 유지
             누워 있거나 RESTING이던 사람은 ②로 나가지 않음 (먼저 일어나야 나갈 수 있음. 2026-11-19 리허설:
              소등하러 레이더 옆을 지나간 사람이 사라진 것을 누워 있던 사람이 나간 것으로 판단해 t3 직후 EMPTY가 됨)
    누워 있던 사람은 "나간 흔적"이 있어야 나간 것으로 봄 (설정에 문 위치가 있을 때, 2026-11-19 사용자 확정):
          CLAUDE.md 원칙 "문 쪽으로 나간 움직임이 없으면 같은 사람이 계속 있다 (방에 1명일 때만)"
          - 나간 흔적 = LD2450이 방 안쪽(문에서 1m 밖)에서 처음 잡은 움직이는 대상이 문 근처(1m 안)에서 사라짐
          - 문 근처에서 처음 잡혀 문 근처에서 사라진 대상 = 들어왔다 나간 다른 사람 → 누운 사람과 무관
          - 나간 흔적이 없으면 어둠 속에서 센서가 다 놓쳐도 OCCUPIED 0.6 유지 (실측: 2.2m에 1분 가만히 누우면 LD2410C·
            LD2450·침대 칸 모두 빈 방과 구분 안 됨 → 센서만으로는 판단 불가. 신뢰도를 낮춰 드러냄, R-13)
          - 밝은데 카메라가 BRIGHT_EMPTY_SECONDS 동안 아무도 못 보면 기억을 지움 (밝으면 누운 사람도 카메라가 봄)
          - 카메라가 일어난 자세를 봐도 기억은 남김 (어둑할 때 지나가는 사람만 보일 수 있음). 나간 흔적으로 지움
    침대 칸 (LD2410C 엔지니어링 모드, 2026-11-19 시연 공간 실측):
          카메라가 누운 자세를 볼 때 LD2450 거리 / 0.75m 반올림 = 침대 칸 (LD2410C 칸 n은 n × 0.75m 근처)
          그 칸 정지 에너지의 10초 중간값 ≥ bed_still_min이면 "침대에 사람 있음" (누움 8~25, 빈 방 3~5)
          - 칸 정보가 있으면 "아직 누워 있음"은 침대 칸으로만 판단: 어둠 속 큰 움직임(다른 사람이 지나감일 수 있음)으로
            일어났다고 보지 않음. 2026-11-19 실측: 지나간 사람의 움직임으로 RESTING이 풀린 뒤 ②가 작동해 EMPTY가 됨
          - LD2410C "없음"·LD2450도 놓쳤어도 누워 있던 사람의 침대 칸이 차 있으면 유지 (실측: 누워서 30초쯤 지나면
            LD2410C가 "없음"이 되고, 지나간 사람이 나간 뒤 LD2450은 누운 사람을 다시 못 잡음)
          - 카메라가 일어난 자세만 보면 침대 기억을 지움. 단 침대 칸이 차 있으면 지우지 않음 (어둑할 때 카메라가
            지나가는 사람만 보고 누운 사람은 못 보는 경우)
          - 칸 정보가 없으면(엔지니어링 모드 꺼짐) 이전 규칙: 누워 있거나 RESTING이면 ②로 나가지 않음
            (어둠 속 ACTIVE 3초면 RESTING이 풀려 다시 ② 가능)
    신뢰도: 카메라+LD2410C 0.95 / 레이더로 유지 0.8~0.85 / 카메라만 0.8 / 비어 있음 0.9
            LD2410C만 "있음"인데 시작 조건이 안 됨(뒤쪽일 수 있음) → EMPTY 0.6 / 걸어 나감 → EMPTY 0.85
            LD2410C "없음"인데 LD2450이 이미 있던 사람을 보거나 침대 칸이 차 있음 → OCCUPIED 0.7
            누워 있던 사람의 나간 흔적이 없어서 유지 (센서는 못 봄) → OCCUPIED 0.6
    LD2410C 연결 안 됨 → LD2450 0.6 / 밝은데 카메라·LD2450 모두 없음 EMPTY 0.7 / 그 밖 UNKNOWN

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
EXIT_SECONDS = 5.0  # 움직이던 사람을 LD2450이 놓친 뒤 이 시간 동안 못 보면 "나감"
MOVING_BEFORE_SECONDS = 3.0  # 사라지기 직전 이 시간 안에 움직였으면 "움직이던 사람" (3초 중간값이 늦게 따라와서)
WALKING_SPEED_CMS = 20       # LD2450 속도가 이 이상이면 걷는 중
GATE_M = 0.75                # LD2410C 거리 칸 하나
LD2450_HOLD_SECONDS = 3.0    # LD2410C "없음"일 때 LD2450이 가만히 있던 사람을 이 시간 안으로 놓친 건 유지
BRIGHT_EMPTY_SECONDS = 10.0  # 밝은데 카메라가 이 시간 동안 아무도 못 보면 누워 있던 기억을 지움


@dataclass
class _Carried:
    identity: str = IDENTITY_UNKNOWN
    identity_confidence: float = 0.0
    activity: Activity = Activity.UNKNOWN
    activity_confidence: float = 0.0


class SingleRoomFusion:
    def __init__(self, bed_still_min: float = 8):
        self.bed_still_min = bed_still_min
        self._bed_range: float | None = None    # 카메라가 누운 자세를 볼 때 LD2450 거리 (침대 칸)
        self._id = 0
        self._present = False
        self._carried: _Carried | None = None
        self._active_since: float | None = None
        self._emitted = False  # 이번 재실 중 사람을 한 번이라도 보냄
        self._seen_at: float | None = None     # 카메라나 LD2450이 마지막으로 사람을 본 시각
        self._moving_at: float | None = None   # 보이는 동안 마지막으로 움직이던(ACTIVE·WALKING·걷는 속도) 시각
        self._in_bed = False                    # 카메라로 마지막에 본 자세가 LYING이거나 RESTING
        self._walked_out = False                # 걸어 나감을 확인한 뒤 아직 아무도 다시 안 들어옴
        self._bed_since: float | None = None    # 누운 것을 처음 본 시각 (이후의 나간 흔적만 셈)
        self._cam_empty_since: float | None = None  # 밝은데 카메라가 아무도 못 본 시작 시각

    def fuse(self, camera: list[VisionPerson] | None, light: Light, radar: RadarSummary, now: float) -> dict:
        """RoomSnapshot의 occupancy·persons 부분 (dict)."""
        cam_seen = bool(camera)
        r10 = radar.ld2410_present if radar.ld2410_available else None
        r50 = radar.ld2450_target if radar.ld2450_available else None

        bed_occupied = self.bed_gate_energy(radar) >= self.bed_still_min
        if cam_seen:
            if any(p.posture == "LYING" or p.activity == "RESTING" for p in camera):
                if not self._in_bed:
                    self._bed_since = now
                self._in_bed = True
                if radar.target_range_m is not None:
                    self._bed_range = radar.target_range_m
            elif not radar.door_known and not (self._in_bed and bed_occupied):
                self._in_bed, self._bed_range = False, None
        if radar.door_known and self._in_bed:
            ex = radar.last_exit
            if ex is not None and ex[0] > (self._bed_since or 0) and not ex[1] and ex[2]:
                self._forget_bed()                  # 방 안쪽에서 시작해 문 근처에서 사라짐 = 나감
            bright_empty = camera is not None and light == Light.BRIGHT and not cam_seen
            self._cam_empty_since = (self._cam_empty_since or now) if bright_empty else None
            if self._cam_empty_since is not None and now - self._cam_empty_since >= BRIGHT_EMPTY_SECONDS:
                self._forget_bed()
        if cam_seen or r50:
            walking = cam_seen and any(p.posture == "WALKING" for p in camera)
            self._seen_at = now
            if radar.motion == Motion.ACTIVE or walking or radar.recent_speed_cms >= WALKING_SPEED_CMS:
                self._moving_at = now
        seen_moving = self._seen_at is not None and self._moving_at is not None \
            and self._seen_at - self._moving_at <= MOVING_BEFORE_SECONDS
        occupancy = self._occupancy(cam_seen, camera is not None, light, r10, r50)
        if occupancy["state"] == "EMPTY" and self._present and not cam_seen and r10 is False:
            # LD2410C가 누운 사람을 못 잡는 거리·자세 → LD2450이 보면(또는 가만히 있던 사람을 잠깐 놓치면) 유지
            briefly_lost = self._seen_at is not None and now - self._seen_at < LD2450_HOLD_SECONDS and not seen_moving
            if r50 or briefly_lost or (self._in_bed and bed_occupied):
                occupancy = {"state": "OCCUPIED", "confidence": 0.7}
            elif self._in_bed and radar.door_known:
                occupancy = {"state": "OCCUPIED", "confidence": 0.6}   # 나간 흔적 없음 (센서는 못 봄)
        if occupancy["state"] == "OCCUPIED" and not self._present and not cam_seen and r10 is not None:
            if not (r50 and r10):
                # LD2410C 혼자: 레이더 뒤쪽일 수 있어 시작 안 함 (걸어 나감을 확인한 뒤라면 더 확실히 비어 있음)
                occupancy = {"state": "EMPTY", "confidence": 0.85 if self._walked_out else 0.6}
        elif occupancy["state"] == "OCCUPIED" and self._present and not cam_seen and not r50 \
                and seen_moving and now - self._seen_at >= EXIT_SECONDS and not self._still_in_bed(radar):
            occupancy = {"state": "EMPTY", "confidence": 0.85}  # 걸어 나감 (LD2410C "있음"은 뒤쪽일 수 있음)
            self._walked_out = True
        if occupancy["state"] != "OCCUPIED":
            self._present, self._carried, self._active_since, self._emitted = False, None, None, False
            self._seen_at, self._moving_at, self._in_bed, self._bed_range = None, None, False, None
            self._bed_since, self._cam_empty_since = None, None
            return {"occupancy": occupancy, "persons": []}

        if not self._present:  # 새로 들어온 사람
            self._present, self._carried, self._walked_out = True, None, False
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

    def _still_in_bed(self, radar: RadarSummary) -> bool:
        """누워 있던 사람이 아직 침대에 있음. 칸 정보가 있으면 침대 칸으로, 없으면 이전 규칙
        (누워 있거나 RESTING. 어둠 속에서 크게 움직이면 RESTING이 풀리며 False)."""
        resting = self._carried is not None and self._carried.activity == Activity.RESTING
        if radar.door_known:
            return self._in_bed                     # 나간 흔적이 나오거나 밝은 빈 방이 확인될 때까지
        energy = self.bed_gate_energy(radar)
        if energy >= 0:
            return (self._in_bed or resting) and energy >= self.bed_still_min
        if resting:
            return True
        return self._in_bed and (self._carried is None or self._carried.activity != Activity.UNKNOWN)

    def _forget_bed(self) -> None:
        self._in_bed, self._bed_range, self._bed_since, self._cam_empty_since = False, None, None, None
        if self._carried is not None and self._carried.activity == Activity.RESTING:
            self._carried.activity, self._carried.activity_confidence = Activity.UNKNOWN, 0.0

    def bed_gate_energy(self, radar: RadarSummary) -> float:
        """기억한 침대 칸의 정지 에너지 중간값. 모르면 -1."""
        if self._bed_range is None or not radar.still_gates:
            return -1
        i = round(self._bed_range / GATE_M)
        return radar.still_gates[i] if i < len(radar.still_gates) else -1

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
