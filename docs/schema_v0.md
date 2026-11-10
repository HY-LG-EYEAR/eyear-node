# EYEAR Node 메시지 형식 v0.1 (초안)

방 장치(Node)가 보내고 받는 메시지 형식입니다. **합의 전 초안**이므로 바꾸고 싶은 부분은 편하게 말해 주세요.
형식이 바뀌면 `schema_version`을 올리고 알려 드립니다.

- 형식 정의(JSON Schema): `schema/json/*.schema.json`
- 예시 메시지: `schema/examples/*.json`
- 가짜 송신기: 실제 장치 없이 같은 형식의 메시지를 MQTT로 보냅니다 (아래 "가짜 송신기 실행").

## 1. 한눈에 보기

```
Node ──(방 상태, 1초마다 + 바뀌면 즉시)──▶ eyear/v0/{room_id}/state          ──▶ 이효창
Node ──(질문에 대한 대답 텍스트)──────────▶ eyear/v0/{room_id}/voice/reply    ──▶ 송정빈(중앙)
Node ──("하이 엘지" 뒤 발화 텍스트)────────▶ eyear/v0/{room_id}/voice/command  ──▶ 송정빈(중앙)
중앙 ──(Node 스피커로 말할 문장)──────────▶ eyear/v0/{room_id}/speak          ──▶ Node
```

- 원본 영상·음성은 보내지 않습니다. 의미 정보(누가, 어떤 자세, 어두움 등)와 인식한 텍스트만 보냅니다.
- 방 ID: `room1`, `room2`, `living`, `laundry`. 실물 Node는 `room1`뿐이고 나머지는 가짜 송신기가 채웁니다.
- 모든 값은 JSON, 시각은 시간대가 포함된 ISO 8601 (`2026-10-30T22:00:00+09:00`).

## 2. 방 상태 메시지 (`.../state`)

**매 메시지가 방 전체 스냅샷**입니다. 메시지 하나만 받아도 방 상태를 알 수 있고, 중간에 몇 개를 놓쳐도 다음 메시지로 복구됩니다.

```json
{
  "schema_version": "0.1",
  "event_id": "6f1c2a7e-0d3b-4b8e-9a51-2c4d8e9f0a11",
  "node_id": "node_room1",
  "room_id": "room1",
  "timestamp": "2026-10-30T22:00:40+09:00",
  "seq": 2,
  "event_type": "STATE_CHANGE",
  "changes": [
    { "type": "POSTURE_CHANGED", "track_id": 1 },
    { "type": "ACTIVITY_CHANGED", "track_id": 1 }
  ],
  "occupancy": { "state": "OCCUPIED", "confidence": 0.97 },
  "persons": [
    {
      "track_id": 1,
      "identity": "MEMBER_A",
      "identity_confidence": 0.92,
      "identity_source": "FACE",
      "posture": "LYING",
      "posture_confidence": 0.84,
      "motion": "SLIGHT",
      "activity": "RESTING",
      "activity_confidence": 0.80,
      "position": { "x": 2.6, "y": 3.1 }
    }
  ],
  "audio": { "context": "SILENT", "confidence": 0.80 },
  "light": "BRIGHT"
}
```

| 필드 | 의미 |
|---|---|
| `event_id` | 메시지마다 새 UUID. 중복 수신 걸러낼 때 사용 |
| `seq` | Node별 일련번호, 1씩 증가. 번호가 빠지면 유실, 갑자기 작아지면 Node 재시작 |
| `event_type` | `HEARTBEAT`(변화 없음, 1초마다) / `STATE_CHANGE`(무언가 바뀜, 즉시) |
| `changes` | 바뀐 것 목록. HEARTBEAT면 빈 리스트 |
| `occupancy` | **방 단위 재실.** 사람이 있는지만 볼 때(로봇청소기, 세탁기)는 이것만 보면 됨 |
| `persons` | 위치·신원을 추적 중인 사람들. 방을 나간 사람은 빠짐 |
| `track_id` | Node 안에서 사람을 구분하는 번호. **방마다 따로 매김** (거실의 1번 ≠ 방1의 1번) |
| `identity` | `MEMBER_A`~`MEMBER_D`(가족), `UNKNOWN`(판단 불가), `GUEST`(미등록 인물) |
| `identity_source` | `FACE`(얼굴), `VOICE`(목소리), `TRACK_CARRIED`(얼굴로 확인한 뒤 어둠 속에서 레이더로 이어받음), `UNKNOWN` |
| `posture` | `STANDING` `SITTING` `LYING` `WALKING` `UNKNOWN` |
| `motion` | 레이더 기준 움직임: `STILL` `SLIGHT` `ACTIVE` `UNKNOWN` |
| `activity` | `RESTING`(휴식, 잠과 구분하지 않음) `OTHER` `UNKNOWN` |
| `position` | 방 기준 좌표(미터). 레이더가 위치를 모르면 `null` |
| `audio.context` | `SILENT` `SPEECH` `CONVERSATION` `MUSIC` `TV` `LOUD_NOISE` `UNKNOWN` |
| `light` | `DARK` `DIM` `BRIGHT` `UNKNOWN` |
| `*_confidence` | 신뢰도 0~1 |

### changes 종류

| type | 단위 | 언제 |
|---|---|---|
| `PERSON_ENTERED` / `PERSON_LEFT` | 사람 (`track_id` 있음) | 들어옴 / 나감 |
| `IDENTITY_CHANGED` | 사람 | 신원 또는 신원 출처가 바뀜 (예: 소등 후 `FACE` → `TRACK_CARRIED`) |
| `POSTURE_CHANGED` | 사람 | 자세가 바뀜 |
| `ACTIVITY_CHANGED` | 사람 | 예: `OTHER` → `RESTING` (중앙이 수면 질문을 할 계기) |
| `OCCUPANCY_CHANGED` | 방 (`track_id: null`) | 재실 상태가 바뀜 |
| `AUDIO_CHANGED` / `LIGHT_CHANGED` | 방 | 소리 환경 / 밝기가 바뀜 |

신뢰도나 위치, `motion`만 바뀐 경우는 변화로 치지 않습니다 (heartbeat에 최신 값이 담김).

### 알아 두면 좋은 상황

- **`occupancy: OCCUPIED` + `persons: []`**: 가만히 누운 사람을 위치 레이더가 놓쳤지만, 재실 레이더는 사람이 있다고 판단한 상태입니다. "사람 있음"으로 처리해 주세요.
- **`UNKNOWN`**: 센서가 없거나 판단할 수 없다는 뜻입니다. 빈 방이나 특정 값으로 추측하지 말아 주세요. 예: `occupancy.state: UNKNOWN`은 "빈 방"이 아닙니다.
- 어두울 때는 신원이 `TRACK_CARRIED`가 되고 신뢰도가 낮아집니다. 불이 켜지면 다시 `FACE`로 돌아옵니다.
- **`activity: RESTING`**: 카메라 기준으로 누운 자세(`LYING`)이고, 이미지–문장 매칭 모델(MobileCLIP)도 "침대에서 쉬는 중"이라고 판단하는 상태가 **5초 이상** 이어질 때 보냅니다.
  - 잠과 휴식은 구분하지 않습니다. 수면 여부는 중앙이 음성으로 물어봐 주세요.
  - 일어난 자세가 **2초 이상** 이어질 때만 `OTHER`로 바뀝니다. 뒤척임이나 이불 때문에 잠깐 자세를 놓쳐도 `RESTING`이 깜빡이지 않습니다.
    따라서 `ACTIVITY_CHANGED`로 `RESTING`이 새로 들어오는 것은 "새로운 휴식 1회"로 봐도 됩니다.
  - 서 있음·앉음·걷기는 `OTHER`입니다. 누워 있지만 두 판단이 엇갈리면 `UNKNOWN`입니다.
  - 불이 꺼진 뒤 휴식 상태를 이어 가는 것은 레이더가 붙은 뒤(M5·M6)에 합니다.
- **`audio.context`**: 1초마다 소리 종류를 판단하고 최근 3초의 다수결을 보냅니다.
  - 사전 학습 모델(YAMNet)은 말하는 사람 수를 세지 못해서, 대화도 대부분 `SPEECH`로 나옵니다. TV 소리도 `SPEECH`나 `MUSIC`으로 나오기 쉽습니다.
  - 키보드 소리처럼 어느 종류에도 해당하지 않으면 `UNKNOWN`입니다.
  - Node가 스피커로 말하는 동안에는 자기 목소리를 분류하지 않고 직전 값을 유지합니다.

### JSON Schema로는 표현되지 않는 규칙 (보내는 쪽이 지킴)

- `HEARTBEAT`면 `changes`가 비어 있고, `STATE_CHANGE`면 1개 이상 있음
- 사람 단위 change에는 `track_id`가 있고, 방 단위 change에는 `track_id: null`
- `persons`에 사람이 있으면 `occupancy.state`는 `EMPTY`가 아님
- `identity`가 `UNKNOWN`이면 `identity_source`도 `UNKNOWN` (그 반대도)
- 한 메시지 안에서 `track_id`는 중복되지 않음

## 3. 음성 메시지

### 중앙 → Node: 말할 문장 (`.../speak`)

```json
{
  "schema_version": "0.1",
  "command_id": "00000000-0000-4000-8000-0000000000c1",
  "timestamp": "2026-10-30T22:00:42+09:00",
  "text": "A님, 주무실 건가요?",
  "expect_reply": true,
  "listen_seconds": 5
}
```

- 중앙은 음성이 아니라 **텍스트**를 보내고, Node가 스피커로 읽습니다.
- `expect_reply: true`면 Node가 말을 끝낸 뒤 최대 `listen_seconds`초(최대 10초) 동안 듣고 `voice/reply`로 대답을 보냅니다.
- 대답이 필요 없는 안내("네, 불을 켤게요")는 `expect_reply: false`로 보내 주세요.
- `command_id`는 중앙이 만듭니다(UUID). 대답에 그대로 돌아옵니다.

### Node → 중앙: 질문에 대한 대답 (`.../voice/reply`)

```json
{ "schema_version": "0.1", "event_id": "...", "node_id": "node_room1", "room_id": "room1",
  "timestamp": "...", "type": "VOICE_REPLY",
  "command_id": "00000000-0000-4000-8000-0000000000c1",
  "text": "응", "confidence": 0.72, "no_response": false }
```

- 응답이 없으면 `text: null`, `confidence: null`, `no_response: true`입니다.
- Node는 들은 말을 텍스트로 옮기기만 합니다. "응/네/어/그래"가 긍정인지는 **중앙이 해석**합니다.
- 실측 결과(2026-11-06, Whisper small) 해석할 때 주의할 점:
  - 인식기가 억양에 따라 **문장부호를 붙입니다**: `응?`, `그래?`, `아니.` → 문장부호는 무시하고 해석해 주세요.
  - 짧은 대답은 **신뢰도가 낮게** 나옵니다(`응?` 0.29, `그래?` 0.51). 0.3 안팎도 정상적인 대답이므로 기준을 너무 높게 잡지 말아 주세요.
  - 기침이나 잡음은 대답으로 보내지 않고 계속 듣습니다. `listen_seconds` 안에 말이 없으면 `no_response`를 보냅니다.
  - 대답을 듣는 중에는 "하이 엘지"를 감지하지 않습니다. 이때 들린 말은 모두 대답으로 보냅니다.

### Node → 중앙: "하이 엘지" 뒤 발화 (`.../voice/command`)

```json
{ "schema_version": "0.1", "event_id": "...", "node_id": "node_room1", "room_id": "room1",
  "timestamp": "...", "type": "VOICE_COMMAND", "text": "불 켜줘", "confidence": 0.81 }
```

- `text`에는 웨이크워드를 뺀 문장만 들어 있습니다.
- "하이 엘지, 불 켜줘"처럼 한 번에 말해도 되고, "하이 엘지"만 말한 뒤 5초 안에 명령을 말해도 됩니다.
- "하이 엘지"가 없는 말은 Node 안에서 버리고 아무것도 보내지 않습니다.
- 누가 말했는지는 담지 않습니다 (방에 1명이라고 가정). 필요하면 같은 방 상태 메시지의 `persons`를 보세요.

## 4. 가짜 송신기 실행

준비 (macOS 기준, 한 번만):

```bash
brew install mosquitto
# Homebrew 설치본에는 설정 파일이 없어서 서비스가 바로 꺼짐 → 최소 설정 파일 생성 (이 노트북 안에서만 접속 허용)
printf 'listener 1883 localhost\nallow_anonymous true\n' > /opt/homebrew/etc/mosquitto/mosquitto.conf
brew services start mosquitto        # 로컬 MQTT 브로커 (localhost:1883)
brew services list | grep mosquitto  # "started"인지 확인 ("error"면 위 설정 파일 확인)
python3.11 -m venv ~/.venvs/eyear-node
~/.venvs/eyear-node/bin/pip install -e ".[dev]"
```

실행 (저장소 폴더에서):

```bash
PY=~/.venvs/eyear-node/bin/python
$PY -m publisher.fake_sender --mqtt localhost              # 시나리오 t0~t7을 5초 간격으로 재생
$PY -m publisher.fake_sender --mqtt localhost --step       # Enter를 누를 때마다 다음 단계
$PY -m publisher.fake_sender --mqtt localhost --reply none # 질문에 대답하지 않음
$PY -m publisher.fake_sender --mqtt localhost --mode random --seed 1   # 무작위 상태 변화 (견고성 테스트)
$PY -m publisher.fake_sender --jsonl out/run.jsonl         # MQTT 없이 파일로만 기록
```

받는 쪽 확인:

```bash
mosquitto_sub -t 'eyear/#' -v
```

- 시나리오 단계마다 화면에 "기대: ..."로 중앙이 할 일이 표시됩니다. 시나리오 내용은 `publisher/scenarios/summer_evening.yaml`에서 읽고 고칠 수 있습니다.
- `--mqtt`를 주면 가짜 음성 Node도 함께 돕니다. 중앙이 `eyear/v0/room1/speak`로 질문을 보내면 1.5초 뒤 `--reply` 값(기본 "응")으로 대답합니다.
- 중앙 없이 질문을 흉내 내 보려면:
  `mosquitto_pub -t eyear/v0/room1/speak -f schema/examples/speak_command_question.json`

## 5. 실제 Node 실행 (신반석 노트북, 방1)

카메라·마이크·스피커가 있는 노트북에서 실제 센서로 같은 형식의 메시지를 보냅니다.

```bash
~/.venvs/eyear-node/bin/pip install -e ".[dev,vision,audio,resting]"   # 처음 한 번
PY=~/.venvs/eyear-node/bin/python
$PY main.py --mqtt localhost               # 카메라 창 + 마이크 + 음성 대화 (room1)
$PY main.py --mqtt localhost --no-camera   # 마이크만 (재실·신원·밝기 = UNKNOWN)
$PY main.py --mqtt localhost --no-audio    # 카메라만 (소리 = UNKNOWN, 음성 대화 없음)
$PY main.py --mqtt localhost --no-resting  # 휴식 판단(MobileCLIP) 끔 → 누움은 activity UNKNOWN
```

- 모델 파일은 `data/models/`에 따로 받아야 합니다 (`pose_landmarker_full.task`, `yamnet.tflite`, 없으면 받는 명령을 알려 줌).
- 센서를 끄거나, 센서 결과가 3초 넘게 오지 않으면 그 값은 `UNKNOWN`으로 나갑니다.

## 6. 합의가 필요한 것

| 항목 | 현재 초안 | 상대 |
|---|---|---|
| 토픽 이름 | `eyear/v0/{room_id}/state` 등 (1절) | 이효창, 송정빈 |
| 방 ID | `room1` `room2` `living` `laundry` | 이효창 |
| 좌표 원점·축 방향 | 방마다 문 쪽 모서리가 원점(임시) | 이효창 |
| heartbeat 주기 | 1초 | 이효창 |
| 가족 이름 표기 | `MEMBER_A`~`MEMBER_D` (실제 이름으로 바꿀지) | 전원 |
| 음성 메시지 필드·듣기 시간 | 3절, 기본 5초 | 송정빈 |
