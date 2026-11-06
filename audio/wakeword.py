"""웨이크워드 "하이 엘지": 음성 인식 텍스트에서 찾는다 (2026-11-06 결정, 학습 불필요).

Whisper가 "하이 LG", "high lg", "하이예지"처럼 여러 형태로 적기 때문에 변형 목록(config/audio.yaml)과 비교한다.
전용 웨이크워드 모델로 바꿀 때는 같은 match() 인터페이스로 교체하면 된다 (R-05).
"""

from audio.stt import normalize

_LEADING = " ,.!?~·、，"


class TextWakeWord:
    def __init__(self, phrases: list[str]):
        self.phrases = sorted({normalize(p) for p in phrases if normalize(p)}, key=len, reverse=True)

    def match(self, text: str) -> str | None:
        """웨이크워드가 없으면 None. 있으면 그 뒤 문장 (웨이크워드만 말했으면 빈 문자열).

        "하이 LG 불 켜줘" → "불 켜줘"
        """
        norm = normalize(text)
        for phrase in self.phrases:
            start = norm.find(phrase)
            if start < 0:
                continue
            # 정규화한 위치를 원문 위치로 되돌려, 뒤 문장을 원문 그대로 돌려준다
            end = start + len(phrase)
            seen = 0
            for i, ch in enumerate(text):
                seen += len(normalize(ch))
                if seen >= end:
                    return text[i + 1:].lstrip(_LEADING).strip()
            return ""
        return None
