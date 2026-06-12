from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chem_evolve_agent.cli import main as run_cli
from scripts.inspect_result_zip import inspect


def _resolve_saisdata_dir(root: Path) -> Path:
    direct_targets = [root / f"target{i}.pdb" for i in range(1, 4)]
    if all(path.exists() for path in direct_targets):
        return root
    nested = root / "37"
    nested_targets = [nested / f"target{i}.pdb" for i in range(1, 4)]
    if all(path.exists() for path in nested_targets):
        return nested
    return root


def main() -> None:
    os.chdir(ROOT)
    saisdata = _resolve_saisdata_dir(Path(os.getenv("SAISDATA_DIR", "/saisdata")))
    saisresult = Path(os.getenv("SAISRESULT_DIR", "/saisresult"))
    top_k = os.getenv("AGENT_TOP_K", "10")
    targets = [
        saisdata / "target1.pdb",
        saisdata / "target2.pdb",
        saisdata / "target3.pdb",
    ]

    sys.argv = [
        "chem_evolve_agent",
        "--targets",
        *(str(target) for target in targets),
        "--out",
        str(saisresult),
        "--rounds",
        os.getenv("AGENT_ROUNDS", "8"),
        "--per-round",
        os.getenv("AGENT_PER_ROUND", "32"),
        "--top-k",
        top_k,
        "--mode",
        os.getenv("AGENT_MODE", "competition"),
        "--docking-limit",
        os.getenv("AGENT_DOCKING_LIMIT", top_k),
        "--runner",
        "agent",
        "--run-seed",
        os.getenv("CHEM_EVOLVE_RUN_SEED", "0"),
    ]
    run_cli()
    inspect(saisresult / "result.zip", ["result1.csv", "result2.csv", "result3.csv"])


if __name__ == "__main__":
    main()
