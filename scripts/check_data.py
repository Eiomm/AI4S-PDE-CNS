from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    _load_dotenv()
    checks: list[tuple[str, bool, str]] = []
    checks.append(_exists("competition_doc", ROOT / "docs/competition_race5_description.md"))
    checks.append(_exists("scoring_doc", ROOT / "docs/sota_tools_and_scoring.md"))
    checks.append(_exists("data_readme", ROOT / "data/README.md"))
    checks.append(_exists("local_smoke_target", ROOT / "examples/target.pdb"))
    checks.extend(_submission_artifact_checks())
    checks.extend(_benchmark_checks(ROOT / "data/benchmarks/manifest.yaml"))
    checks.append(_benchmark_prior_check(ROOT / "data/benchmarks/benchmark_prior.json"))

    config = Path(os.getenv("AIZYNTHFINDER_CONFIG", ROOT / "data/aizynthfinder/config.yml"))
    checks.append(_exists("aizynthfinder_config", config))
    checks.append(_aizynthfinder_config_portable(config))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        status = "通过" if ok else "缺失"
        if not ok:
            failed += 1
        print(f"{status:<8} {name:<{width}}  {detail}")
    if failed:
        print(f"\nDATA_CHECK_FAILED 数据检查失败：缺失 {failed} 项")
        return 1
    print("\nDATA_CHECK_OK 数据检查通过")
    return 0


def _exists(name: str, path: Path) -> tuple[str, bool, str]:
    return name, path.exists(), str(path)


def _submission_artifact_checks() -> list[tuple[str, bool, str]]:
    return [
        _exists("submission_entrypoint", ROOT / "Code/main.py"),
        _text_contains("submission_entrypoint:targets", ROOT / "Code/main.py", ["target1.pdb", "target2.pdb", "target3.pdb"]),
        _text_contains("submission_entrypoint:inspect", ROOT / "Code/main.py", ["result1.csv", "result2.csv", "result3.csv"]),
        _exists("submission_readme", ROOT / "Code/README.md"),
        _exists("submission_run_script", ROOT / "run.sh"),
        _text_contains("submission_run_script:main", ROOT / "run.sh", ["Code/main.py"]),
        _text_contains("submission_run_script:docker_llm_env", ROOT / "run.sh", ["configs/docker_llm.env"]),
        _exists("submission_docker_build_script", ROOT / "docker_build.sh"),
        _text_contains("submission_docker_build_script:registry", ROOT / "docker_build.sh", ["DOCKER_REGISTRY", "scripts/docker_build_push.sh"]),
        _exists("docker_llm_env", ROOT / "configs/docker_llm.env"),
        _text_not_contains("docker_llm_env:no_placeholder", ROOT / "configs/docker_llm.env", ["<your-docker-llm-api-key>"]),
        _exists("dockerfile", ROOT / "Dockerfile"),
        _text_contains("dockerfile:entrypoint", ROOT / "Dockerfile", ["/app/run.sh"]),
        _text_contains("dockerfile:training_code", ROOT / "Dockerfile", ["/app/training_code"]),
        _exists("training_code_readme", ROOT / "app/training_code/README.md"),
        _exists("training_code_prior_script", ROOT / "app/training_code/build_benchmark_prior.py"),
    ]


def _text_contains(name: str, path: Path, needles: list[str]) -> tuple[str, bool, str]:
    if not path.exists():
        return name, False, str(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        return name, False, f"{path} (missing: {', '.join(missing)})"
    return name, True, str(path)


def _text_not_contains(name: str, path: Path, needles: list[str]) -> tuple[str, bool, str]:
    if not path.exists():
        return name, False, str(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    present = [needle for needle in needles if needle in text]
    if present:
        return name, False, f"{path} (unexpected: {', '.join(present)})"
    return name, True, str(path)


def _benchmark_checks(manifest_path: Path) -> list[tuple[str, bool, str]]:
    checks = [_exists("benchmark_manifest", manifest_path)]
    if not manifest_path.exists():
        return checks
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("root", manifest_path.parent))
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    for benchmark in payload.get("benchmarks", []):
        for target in benchmark.get("targets", []):
            target_id = target.get("id", "unknown")
            for key in ("target_pdb", "actives_smi", "decoys_smi", "metadata_json"):
                value = target.get(key)
                if value:
                    checks.append(_exists(f"benchmark:{target_id}:{key}", root / value))
    return checks


def _benchmark_prior_check(path: Path) -> tuple[str, bool, str]:
    if not path.exists():
        return "benchmark_prior", False, str(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "benchmark_prior", False, f"{path} (JSON 无效)"
    ok = (
        isinstance(payload.get("active_summary"), dict)
        and isinstance(payload.get("decoy_summary"), dict)
        and "smiles" not in json.dumps(payload).lower()
    )
    detail = str(path) if ok else f"{path} (只能包含聚合摘要，不能包含 SMILES)"
    return "benchmark_prior", ok, detail


def _aizynthfinder_config_portable(path: Path) -> tuple[str, bool, str]:
    if not path.exists():
        return "aizynthfinder_config:portable", False, str(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    bad_tokens = ["/data/wangjunao", "/workspace/"]
    bad = [token for token in bad_tokens if token in text]
    if bad:
        return "aizynthfinder_config:portable", False, f"{path} (包含绝对路径：{', '.join(bad)})"
    return "aizynthfinder_config:portable", True, str(path)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT / ".env")


if __name__ == "__main__":
    raise SystemExit(main())
