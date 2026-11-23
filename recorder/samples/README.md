# 재생용 Node 기록

실제 Node(방1)가 보낸 메시지를 `main.py --jsonl`로 저장한 것. 원본 영상·음성은 없고 의미 정보만 있다.
재생: `python -m recorder.replay <파일> --mqtt localhost` (구간만: `--from 초 --to 초`, 빠르게: `--speed 2`).

## 2026-11-19_room1_rest_passerby.jsonl (227초, 시연 공간)

A 역할 MEMBER_D. 노트북·레이더는 config/radar.yaml 설치 위치 기준.

| 시점(초) | 장면 | Node가 보낸 것 |
|---|---|---|
| 0~17 | 방이 빔 | EMPTY |
| 17~52 | A가 들어와 침대에 누움 (밝음) | OCCUPIED, LYING → RESTING (약 38초) |
| 52~76 | 소등 | DARK, 신원 TRACK_CARRIED, OCCUPIED 유지 |
| 76~116 | 다른 사람이 들어와 침대 근처에서 멈췄다 나감 | OCCUPIED·RESTING 유지 (지나간 사람으로 판단) |
| 116~181 | A가 1분 동안 가만히 누워 있음 | OCCUPIED 유지 (0.7~0.85) |
| 181~210 | A가 일어나 방을 나감 | 나간 뒤 약 5초에 EMPTY |
| 210~227 | 불을 켬 | EMPTY, BRIGHT |

음성 메시지는 없음 (이 테스트는 마이크를 끄고 진행).
