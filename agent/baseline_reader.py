from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


BASELINE_REPOS = ("PDEBench", "neuraloperator", "deeponet", "Physics-informed-DeepONets")
ALLOWED_SUFFIXES = {".py", ".ipynb", ".md"}
SKIP_SUFFIXES = {".npy", ".npz", ".mat", ".pt", ".pth", ".tar", ".h5", ".hdf5", ".png", ".jpg", ".jpeg"}
RELEVANT_HINTS = {
    "fno",
    "pinn",
    "deeponet",
    "burger",
    "burgers",
    "unet",
    "train",
    "metrics",
    "config",
    "readme",
}


def _symbols(path: Path, *, limit: int = 8) -> list[str]:
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(node.name)
            if len(names) >= limit:
                break
    return names


def _is_relevant(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return False
    if suffix not in ALLOWED_SUFFIXES:
        return False
    lowered = path.as_posix().lower()
    return any(hint in lowered for hint in RELEVANT_HINTS)


def _agent_use(repo: str, relative_path: str, symbols: list[str]) -> str:
    lowered = f"{repo}/{relative_path}".lower()
    symbol_text = " ".join(symbols).lower()
    if "fno" in lowered or "spectral" in symbol_text:
        return "Study FNO spectral operator architecture, temporal stride, and checkpoint-compatible training choices."
    if "pinn" in lowered or "physics-informed" in repo.lower():
        return "Study physics residual and PI-DeepONet loss ideas for stability-oriented experiments."
    if "deeponet" in lowered:
        return "Study branch/trunk operator-learning structure as a Task1/Task2 model-family candidate."
    if "unet" in lowered:
        return "Study U-Net rollout baseline and compare against FNO-style predictors."
    if "metrics" in lowered:
        return "Study PDEBench metric conventions and validation bookkeeping."
    return "Provide baseline source context for Agent hypothesis and code evolution."


def index_baseline_repos(root: str | Path, *, max_files_per_repo: int = 24) -> dict[str, dict[str, Any]]:
    baseline_root = Path(root)
    result: dict[str, dict[str, Any]] = {}
    if not baseline_root.is_dir():
        return result
    for repo in BASELINE_REPOS:
        repo_dir = baseline_root / repo
        if not repo_dir.is_dir():
            continue
        files: list[dict[str, Any]] = []
        for path in sorted(repo_dir.rglob("*")):
            if len(files) >= max_files_per_repo:
                break
            if not path.is_file() or not _is_relevant(path):
                continue
            relative = path.relative_to(repo_dir).as_posix()
            symbols = _symbols(path)
            files.append(
                {
                    "path": relative,
                    "suffix": path.suffix.lower(),
                    "symbols": symbols,
                    "agent_use": _agent_use(repo, relative, symbols),
                }
            )
        result[repo] = {
            "path": repo_dir.as_posix(),
            "files": files,
        }
    return result


def summarize_baseline_context(root: str | Path, *, max_source_files: int = 8) -> dict[str, dict[str, Any]]:
    index = index_baseline_repos(root)
    context: dict[str, dict[str, Any]] = {}
    for repo, payload in index.items():
        files = list(payload.get("files", []))
        source_files = [f"{repo}/{item['path']}" for item in files[:max_source_files]]
        uses = []
        for item in files:
            use = item.get("agent_use")
            if isinstance(use, str) and use not in uses:
                uses.append(use)
        symbols = [symbol for item in files for symbol in item.get("symbols", [])]
        context[repo] = {
            "repo_path": payload.get("path"),
            "file_count": len(files),
            "source_files": source_files,
            "symbols": symbols[:16],
            "agent_use": uses[:6],
            "summary": (
                f"{repo} baseline context with {len(files)} relevant files: {', '.join(source_files[:4])}. "
                f"Symbols: {', '.join(symbols[:8])}"
            ),
        }
    return context
