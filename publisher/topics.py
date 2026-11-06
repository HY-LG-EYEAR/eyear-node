"""MQTT 토픽 이름 (팀원2·팀원3과 합의 전 초안)."""

PREFIX = "eyear/v0"


def state(room_id: str) -> str:
    """Node → 중앙: 방 상태 (NodeStateMessage)."""
    return f"{PREFIX}/{room_id}/state"


def voice_reply(room_id: str) -> str:
    """Node → 중앙: 질문에 대한 대답 (VoiceReply)."""
    return f"{PREFIX}/{room_id}/voice/reply"


def voice_command(room_id: str) -> str:
    """Node → 중앙: "하이 엘지" 뒤 발화 (VoiceCommand)."""
    return f"{PREFIX}/{room_id}/voice/command"


def speak(room_id: str) -> str:
    """중앙 → Node: 말할 문장 (SpeakCommand)."""
    return f"{PREFIX}/{room_id}/speak"


SPEAK_ALL_ROOMS = f"{PREFIX}/+/speak"


def room_of(topic: str) -> str:
    return topic.split("/")[2]
