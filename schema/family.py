"""가족 목록(config/family.yaml) 읽기."""

import re
from pathlib import Path

import yaml

from schema.enums import RESERVED_IDENTITIES

DEFAULT_FAMILY_PATH = Path(__file__).resolve().parent.parent / "config" / "family.yaml"
_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_family(path: Path | str = DEFAULT_FAMILY_PATH) -> frozenset[str]:
    members = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("members") or []
    for name in members:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"가족 이름은 영문 대문자·숫자·밑줄만 가능: {name!r}")
        if name in RESERVED_IDENTITIES:
            raise ValueError(f"예약어는 가족 이름으로 쓸 수 없음: {name}")
    if len(members) != len(set(members)):
        raise ValueError("가족 이름이 중복됨")
    return frozenset(members)
