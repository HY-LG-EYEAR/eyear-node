# Node 등록 API (앱 → Node, 초안 0.1)

가족 얼굴·목소리를 **앱에서** 등록한다. 앱은 사진·녹음을 **같은 와이파이의 Node로 바로** 보내고,
Node가 특징값만 저장한 뒤 원본은 버린다. 앱 백엔드(EC2)와 클라우드에는 생체정보를 보내지 않는다 (R-10).

- 상태: **초안** (2026-11-27 팀 결정 "A안", 예원과 형식 합의 전). 바뀌면 이 문서와 버전을 올리고 예원에게 알림.
- 구현: `enroll/` (eyear-node). 실행: `python -m enroll.server` (기본 `0.0.0.0:8770`)
- 앱 백엔드 DB에는 **신원 ID, 이름, 선호 온도, 등록 여부만** 둔다. 특징값·사진·녹음은 두지 않는다.

```
앱: 가족 추가 (이름·선호 온도) ─────────────▶ 앱 백엔드 DB (identity, 이름, 온도, 등록 여부)
 └ 사진 5장·녹음 5개 ──(집 와이파이)──▶ Node :8770 ─▶ 특징값만 data/faces, data/voices
                                                      원본은 메모리에서 바로 버림
```

## 연결
- 주소: `http://<Node IP>:8770` — Node가 켜질 때 화면에 주소와 **6자리 연결 코드**를 표시한다.
  시연은 폰과 Node 노트북을 같은 핫스팟에 둔다. 코드를 고정하려면 `EYEAR_PAIRING_CODE=123456 python -m enroll.server`.
- 모든 요청에 헤더 `X-EYEAR-Pairing: <6자리>`. 틀리면 `401 BAD_PAIRING_CODE`.
- 집 안 전용이다. 이 포트를 인터넷(EC2 보안 그룹, 공유기 포트 포워딩)에 열지 않는다.
- 응답은 JSON. 오류는 `{"error": "코드", "message": "설명"}`.

## 신원 ID
- `MEMBER_A`처럼 영문 대문자로 시작, 대문자·숫자·밑줄, 32자 이하. `UNKNOWN`·`GUEST`는 예약어라 쓸 수 없다.
- **앱(백엔드)이 정한다.** Node 메시지의 `persons[].identity`, World Model·중앙의 identity와 같은 값이다.
- 사람 이름("효창")은 Node에 보내지 않는다 (앱 DB에만).

## 요청

### `GET /v1/family` — 등록 현황
```json
{"members": [{"identity": "MEMBER_D", "face_samples": 12, "voice_samples": 5}]}
```

### `PUT /v1/family/{identity}` — 가족 추가
본문 없음. 새로 추가하면 `201`, 이미 있으면 `200`. → `{"identity": "MEMBER_E", "added": true}`

### `POST /v1/family/{identity}/face?mode=replace|append` — 얼굴 사진
- `multipart/form-data`, 파일 필드 이름은 `files` (여러 개, 보낸 순서가 `index`). 한 번에 최대 10장, 본문 40MB.
- 형식: jpg, png (iPhone HEIC는 앱에서 jpg로 바꿔 보낼 것). **한 장에 한 사람**, 정면·좌우·위아래로 조금씩 다르게 5장 권장.
- `mode`: `replace`(기본, 기존 얼굴 등록을 바꿈) / `append`(기존에 더함).
- 쓸 수 있는 사진이 **3장 미만이면 저장하지 않는다** (`saved: false`, 기존 등록 유지). `append`는 1장이어도 더한다.

```json
{"identity": "MEMBER_E", "accepted": 4, "saved": true, "total": 4,
 "rejected": [{"index": 2, "reason": "MULTIPLE_FACES"}]}
```
| reason | 뜻 |
|---|---|
| `NO_FACE` | 얼굴을 찾지 못함 |
| `MULTIPLE_FACES` | 얼굴이 2개 이상 |
| `FACE_TOO_SMALL` | 얼굴이 너무 작음 (더 가까이) |
| `UNREADABLE` | 이미지를 읽을 수 없음 (형식 확인) |

### `POST /v1/family/{identity}/voice?mode=replace|append` — 목소리 녹음
- `multipart/form-data`, 파일 필드 `files`. 형식: m4a(AAC), wav, mp3 등 (Node가 16kHz 모노로 바꿈).
- 한 파일에 문장 하나, **말소리 0.8~15초** (앞뒤 조용한 부분은 Node가 자름). 5개 권장. 저장 기준은 얼굴과 같음(3개).
- 권장 문장 (Node 등록 CLI와 같음): "하이 엘지, 불 켜줘" / "하이 엘지, 에어컨 켜줘" / "오늘 저녁은 뭐 먹을까요" / "방 안이 조금 덥네요" / "내일 아침 일곱 시에 깨워 주세요"
- 응답 형식은 얼굴과 같다. reason: `NO_SPEECH`(소리가 너무 작음), `TOO_SHORT`, `TOO_LONG`, `UNREADABLE`.

### `DELETE /v1/family/{identity}` — 얼굴·목소리 등록 삭제
가족 목록에서는 지우지 않는다 (지난 기록의 identity 검사가 깨지지 않게).
→ `{"identity": "MEMBER_E", "deleted": {"face": true, "voice": true}}`

### 그 밖의 오류
| 상태 | error | 경우 |
|---|---|---|
| 400 | `INVALID_IDENTITY`, `RESERVED_IDENTITY` | 신원 ID 형식 |
| 400 | `NO_FILES`, `TOO_MANY_FILES`, `BAD_MODE`, `BAD_MULTIPART` | 요청 형식 |
| 404 | `UNKNOWN_IDENTITY` | 가족 목록에 없음 → 먼저 `PUT` |
| 413 | `TOO_LARGE` | 본문 40MB 초과 |
| 415 | `UNSUPPORTED_CONTENT_TYPE` | multipart가 아님 |

## Node 쪽 동작
- 실행 중인 Node(`main.py`)는 등록 파일이 바뀌면 **2초 안에 다시 읽는다** (다시 켤 필요 없음). 가족 추가도 반영(추가만).
- 등록 서버와 Node는 별도 프로그램이다 (얼굴·목소리 모델을 각자 불러옴).
- 로그에는 요청 줄만 남기고 사진·녹음·특징값은 남기지 않는다.

## 알려진 한계 (측정 예정)
- 폰 사진·폰 가까이에서 한 녹음은 방 카메라(각도·거리)·방 마이크(2m)와 조건이 달라 정확도가 떨어질 수 있다.
  필요하면 그 방에서 `append`로 추가 등록 (`python -m vision.enroll --name X --append`, `python -m audio.enroll_voice --name X --append`).
- Node가 여러 대인 집: 지금은 Node마다 등록. 나중에 집 안 허브가 특징값을 보관하고 같은 집 Node에 나눠 주는 방식 (클라우드 X).
- 연결 코드 외 인증·암호화(HTTPS) 없음. 집 와이파이 안에서만 쓰는 전제.
