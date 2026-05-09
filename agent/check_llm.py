from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .llm import build_llm_client, logged_completion
from .logging import LLMCallLogger
from .run import load_config


def check_llm(config_path: str | Path, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root else Path.cwd()
    config = load_config(config_path)
    client = build_llm_client(config)
    run_dir = root / "runs" / f"api-check-{client.provider}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = LLMCallLogger(run_dir / "api_check_logs.log")
    response = logged_completion(
        client,
        logger,
        [
            {
                "role": "system",
                "content": "You are a concise connectivity test responder.",
            },
            {
                "role": "user",
                "content": "Reply with exactly: API_OK",
            },
        ],
    )
    (run_dir / "response.json").write_text(
        json.dumps(
            {
                "provider": client.provider,
                "model": client.model,
                "content": response.get("content", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one configured LLM provider.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(check_llm(args.config))


if __name__ == "__main__":
    main()
