from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path) -> dict[str, str]:
    """读取简单的 `.env` 文件，并把尚未存在的变量写入当前进程环境。

    这里不依赖 `python-dotenv`，是为了保持 Task1 运行时足够轻。
    支持最常见的 `KEY=value` 格式；空行和 `#` 注释会被忽略。
    返回值只包含本次从文件解析出的键值，调用方不要打印敏感值。
    """

    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def first_available_env(keys: list[str]) -> str | None:
    """按优先级返回第一个已经设置且非空的环境变量名。"""

    for key in keys:
        if os.environ.get(key):
            return key
    return None
