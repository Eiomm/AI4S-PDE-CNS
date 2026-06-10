from __future__ import annotations

import json
from typing import Any


def json_event(event: str, **fields: Any) -> str:
    return json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True)
