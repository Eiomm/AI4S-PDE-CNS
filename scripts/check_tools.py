#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.runtime_tools import list_tool_specs, load_target, run_sota_sbdd_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 AI4S agent 需要的外部工具是否可用。", add_help=False)
    parser._optionals.title = "参数"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--strict", action="store_true", help="只要有工具缺失就返回非零退出码")
    parser.add_argument("--require-sbdd", action="store_true", help="要求外部 SBDD 生成器已配置，并实际探测一次")
    parser.add_argument("--probe-target", default="examples/target.pdb", help="SBDD 探测使用的靶点 PDB")
    parser.add_argument("--probe-limit", type=int, default=1, help="SBDD 探测要求的最少有效 SMILES 数")
    args = parser.parse_args()
    _load_dotenv()
    print("AI4S 工具可用性检查")
    print("=" * 88)
    missing_required: list[str] = []
    for spec in list_tool_specs():
        status = "可用" if spec.available else "缺失"
        programs = ", ".join(spec.required_programs)
        print(f"{status:<8} {spec.name:<32} {programs}")
        print(f"         用途：{spec.purpose}")
        if spec.note:
            print(f"         备注：{spec.note}")
        if not spec.available and (args.strict or (args.require_sbdd and spec.name == "sota_sbdd_generator_tool")):
            missing_required.append(spec.name)
    if missing_required:
        raise SystemExit("MISSING_REQUIRED_TOOLS: 缺少必需工具：" + ", ".join(missing_required))
    if args.require_sbdd:
        _probe_sbdd(Path(args.probe_target), args.probe_limit)


def _probe_sbdd(target_path: Path, limit: int) -> None:
    if limit <= 0:
        raise SystemExit("SBDD_PROBE_FAILED: --probe-limit 必须是正数")
    try:
        target = load_target(target_path)
        with tempfile.TemporaryDirectory() as tmp:
            generated = run_sota_sbdd_generator(target, Path(tmp), limit=limit)
    except Exception as exc:
        raise SystemExit(f"SBDD_PROBE_FAILED: {exc}") from exc
    if len(generated) < limit:
        raise SystemExit(f"SBDD_PROBE_FAILED: 只生成了 {len(generated)} 个有效 SMILES，期望至少 {limit} 个")
    print(f"PROBE_OK SBDD 生成器探测通过：生成 {len(generated)} 个有效 SMILES")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


if __name__ == "__main__":
    main()
