#!/bin/bash
# 같은 네트워크(휴대폰 핫스팟·와이파이)의 팀원 노트북이 접속할 수 있는 MQTT 브로커.
#
#   scripts/broker_lan.sh          # 아이디·비밀번호 필요 (비밀번호 파일을 먼저 만들 것, 아래 안내)
#   scripts/broker_lan.sh --open   # 비밀번호 없이 (팀만 쓰는 휴대폰 핫스팟에서만)
#
# - 노트북 안에서만 쓰는 brew services의 mosquitto와 같은 1883 포트를 쓰므로, 그것을 먼저 끈다:
#     brew services stop mosquitto      (끝나면 brew services start mosquitto)
# - macOS가 "mosquitto의 연결을 허용할까요?"라고 물으면 허용. 방화벽이 막으면 다른 노트북이 못 붙는다.
# - Node 쪽 아이디·비밀번호: export EYEAR_MQTT_USER=eyear EYEAR_MQTT_PASSWORD=...
#   (비밀번호는 eyear-team 레포에 쓰지 말고 직접 전달)
set -e
DIR="$HOME/.eyear"
PW="$DIR/mosquitto_passwd"
CONF="$DIR/mosquitto_lan.conf"
mkdir -p "$DIR"

if brew services list 2>/dev/null | grep -q "^mosquitto .*started"; then
  echo "brew services의 mosquitto가 1883 포트를 쓰는 중입니다. 먼저 끄세요: brew services stop mosquitto"
  exit 1
fi

{
  echo "listener 1883 0.0.0.0"
  echo "persistence false"
  if [ "$1" = "--open" ]; then
    echo "allow_anonymous true"
  else
    if [ ! -f "$PW" ]; then
      echo "비밀번호 파일이 없습니다. 먼저 만드세요: mosquitto_passwd -c $PW eyear" >&2
      exit 1
    fi
    chmod 600 "$PW"
    echo "allow_anonymous false"
    echo "password_file $PW"
  fi
} > "$CONF"

IP=$(ipconfig getifaddr en0 2>/dev/null || true)
echo "브로커 주소: ${IP:-(IP 확인 실패: 시스템 설정 > 네트워크에서 확인)}:1883"
if [ "$1" = "--open" ]; then echo "비밀번호 없음 (핫스팟에서만 쓸 것)"; else echo "아이디 eyear, 비밀번호는 직접 전달"; fi
echo "다른 노트북에서 확인: mosquitto_sub -h ${IP:-<주소>} -t 'eyear/v0/#' -v"
exec /opt/homebrew/sbin/mosquitto -c "$CONF" -v
