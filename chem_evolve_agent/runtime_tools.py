from __future__ import annotations

import csv
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chem_evolve_agent.chem_ops import binding_score_from_vina_energy, canonicalize_reaction_step, canonicalize_smiles, validate_route
from chem_evolve_agent.models import Route


class ToolError(RuntimeError):
    pass


class ToolSpec(BaseModel):
    name: str
    purpose: str
    required_programs: list[str]
    available: bool
    note: str = ""


@dataclass
class TargetContext:
    target_id: str
    path: Path
    atom_count: int
    residues: list[str]
    pocket_center: tuple[float, float, float]
    pocket_box_size: tuple[float, float, float]
    pocket_summary: str


def json_event(event: str, **payload: Any) -> str:
    return json.dumps({"event": event, **payload}, sort_keys=True, ensure_ascii=False)


def load_target(target_path: Path) -> TargetContext:
    if not target_path.exists():
        raise FileNotFoundError(f"target PDB not found: {target_path}")
    atoms: list[tuple[float, float, float]] = []
    residues: list[str] = []
    for line in target_path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append((x, y, z))
        residue = f"{line[17:20].strip()}:{line[21].strip()}:{line[22:26].strip()}"
        if residue not in residues:
            residues.append(residue)
    if not atoms:
        raise ValueError(f"target PDB contains no parseable atoms: {target_path}")
    xs, ys, zs = zip(*atoms)
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    box = (max(22.0, max(xs) - min(xs) + 6), max(22.0, max(ys) - min(ys) + 6), max(22.0, max(zs) - min(zs) + 6))
    return TargetContext(
        target_id=target_path.stem,
        path=target_path,
        atom_count=len(atoms),
        residues=residues,
        pocket_center=tuple(round(value, 3) for value in center),
        pocket_box_size=tuple(round(value, 3) for value in box),
        pocket_summary=f"pdb_atoms={len(atoms)} residues={len(residues)} center={tuple(round(v, 2) for v in center)} box={tuple(round(v, 2) for v in box)}",
    )


def list_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="rdkit_property_tool",
            purpose="SMILES 合法性、性质、QED、SA 分数",
            required_programs=["python:rdkit"],
            available=has_python_module("rdkit"),
        ),
        ToolSpec(
            name="sota_sbdd_generator_tool",
            purpose="调用 MolCRAFT/DiffGui/DiffSBDD/TargetDiff 风格的外部 pocket-conditioned 3D 生成器",
            required_programs=["AI4S_SBDD_GENERATOR_CMD"],
            available=_sbdd_command_available(),
            note="命令需读取 AI4S_TARGET_PDB/AI4S_POCKET_JSON/AI4S_OUTPUT_SMILES/AI4S_LIMIT。",
        ),
        ToolSpec(
            name="vina_docking_tool",
            purpose="用 AutoDock Vina 产生 binding_score",
            required_programs=["python:vina", "obabel"],
            available=has_python_module("vina") and find_executable("obabel", "babel") is not None,
            note="当前实现使用 Vina Python binding；AutoDock Vina 官方说明 Python binding 和二进制 executable 是分开的。",
        ),
        ToolSpec(
            name="aizynthfinder_route_tool",
            purpose="用 AiZynthFinder 做 MCTS/template 多步逆合成规划",
            required_programs=["aizynthcli", "AIZYNTHFINDER_CONFIG"],
            available=find_executable("aizynthcli", "aizynthfinder") is not None and _configured_path_exists("AIZYNTHFINDER_CONFIG"),
            note="" if _configured_path_exists("AIZYNTHFINDER_CONFIG") else "需要 download_public_data 生成 config.yml；未配置时不会自动兜底。",
        ),
    ]


def run_sota_sbdd_generator(target: TargetContext, out_dir: Path, limit: int) -> list[str]:
    command = os.getenv("AI4S_SBDD_GENERATOR_CMD")
    if not command:
        raise ToolError("AI4S_SBDD_GENERATOR_CMD is required for sota_sbdd_generator_tool")
    out_dir.mkdir(parents=True, exist_ok=True)
    pocket_json = out_dir / "pocket.json"
    output_smiles = out_dir / "sota_sbdd_smiles.csv"
    pocket_json.write_text(
        json.dumps(
            {
                "target_id": target.target_id,
                "center": target.pocket_center,
                "box_size": target.pocket_box_size,
                "summary": target.pocket_summary,
            },
            ensure_ascii=False,
        )
    )
    env = os.environ.copy()
    env.update(
        {
            "AI4S_TARGET_PDB": str(target.path),
            "AI4S_POCKET_JSON": str(pocket_json),
            "AI4S_OUTPUT_SMILES": str(output_smiles),
            "AI4S_LIMIT": str(limit),
        }
    )
    result = run_command(shlex.split(command), timeout=int(os.getenv("AI4S_SBDD_TIMEOUT", "900")), env=env)
    if result.returncode != 0:
        raise ToolError(f"SOTA SBDD generator failed: {result.stderr.strip() or result.stdout.strip()}")
    if not output_smiles.exists():
        raise ToolError(f"SOTA SBDD generator did not write {output_smiles}")
    generated = _validated_generated_smiles(output_smiles, limit)
    if not generated:
        raw_count = len(read_smiles_file(output_smiles))
        raise ToolError(f"SOTA SBDD generator produced no RDKit-valid SMILES: rows={raw_count}")
    return generated


def run_vina_binding_score(smiles: str, target: TargetContext, out_dir: Path) -> tuple[float, float]:
    obabel = require_executable("obabel", "babel")
    try:
        from vina import Vina
    except Exception as exc:
        raise ToolError("python package vina is required for docking mode") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    ligand_smi = out_dir / "ligand.smi"
    ligand_pdbqt = out_dir / "ligand.pdbqt"
    receptor_pdbqt = out_dir / "receptor.pdbqt"
    pose_pdbqt = out_dir / "pose.pdbqt"
    ligand_smi.write_text(smiles + "\n")
    _must_run([obabel, "-ismi", str(ligand_smi), "-opdbqt", "-O", str(ligand_pdbqt), "--gen3d"], timeout=120)
    _must_run([obabel, "-ipdb", str(target.path), "-opdbqt", "-O", str(receptor_pdbqt), "-xr"], timeout=240)
    cx, cy, cz = target.pocket_center
    sx, sy, sz = target.pocket_box_size
    vina = Vina(sf_name="vina", cpu=int(os.getenv("AI4S_VINA_CPU", "1")), verbosity=0)
    vina.set_receptor(str(receptor_pdbqt))
    vina.set_ligand_from_file(str(ligand_pdbqt))
    vina.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])
    vina.dock(exhaustiveness=int(os.getenv("AI4S_VINA_EXHAUSTIVENESS", "8")), n_poses=1)
    energies = vina.energies(n_poses=1)
    if len(energies) == 0:
        raise ToolError("Vina returned no docked poses")
    energy = float(energies[0][0])
    vina.write_poses(str(pose_pdbqt), n_poses=1, overwrite=True)
    return binding_score_from_vina_energy(energy), energy


def run_aizynthfinder_route(smiles: str, out_dir: Path) -> Route:
    aizynthcli = require_executable("aizynthcli")
    config = os.getenv("AIZYNTHFINDER_CONFIG")
    if not config:
        raise ToolError("AIZYNTHFINDER_CONFIG is required for aizynthfinder_route_tool")
    config_path = Path(config)
    if not config_path.exists():
        raise ToolError(f"AIZYNTHFINDER_CONFIG does not exist: {config_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    smiles_file = out_dir / "target.smi"
    output_file = out_dir / "aizynth_output.json.gz"
    smiles_file.write_text(canonicalize_smiles(smiles) + "\n")
    result = run_command(
        [
            aizynthcli,
            "--config",
            str(config_path),
            "--smiles",
            str(smiles_file),
            "--output",
            str(output_file),
        ],
        timeout=int(os.getenv("AIZYNTHFINDER_TIMEOUT", "900")),
    )
    if result.returncode != 0:
        raise ToolError(f"AiZynthFinder failed: {result.stderr.strip() or result.stdout.strip()}")
    if not output_file.exists():
        raise ToolError(f"AiZynthFinder did not write {output_file}")
    row = _read_first_aizynth_row(output_file)
    if not _truthy(row.get("is_solved")):
        raise ToolError("AiZynthFinder found no solved route")
    trees = row.get("trees")
    if isinstance(trees, str):
        trees = json.loads(trees)
    if not isinstance(trees, list) or not trees:
        raise ToolError("AiZynthFinder output contains no route trees")
    return _select_best_aizynth_route(canonicalize_smiles(smiles), trees)


def read_smiles_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv":
        rows = csv.DictReader(text.splitlines())
        smiles: list[str] = []
        for row in rows:
            value = row.get("smiles") or row.get("mol_smiles") or next(iter(row.values()), "")
            if value:
                smiles.append(value.strip())
        return smiles
    return [line.split()[0].strip() for line in text.splitlines() if line.strip()]


def _validated_generated_smiles(path: Path, limit: int) -> list[str]:
    valid: list[str] = []
    seen: set[str] = set()
    for raw in read_smiles_file(path):
        try:
            smiles = canonicalize_smiles(raw)
        except Exception:
            continue
        if smiles in seen:
            continue
        seen.add(smiles)
        valid.append(smiles)
        if len(valid) >= limit:
            break
    return valid


def _read_first_aizynth_row(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list) and payload["data"]:
        row = payload["data"][0]
        if isinstance(row, dict):
            return row
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    raise ToolError(f"Unsupported AiZynthFinder output format: {path}")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _select_best_aizynth_route(target_smiles: str, trees: list[Any]) -> Route:
    target = canonicalize_smiles(target_smiles)
    best_route: Route | None = None
    best_score = -1.0
    rejected: list[str] = []
    for tree in trees:
        try:
            route = _route_from_aizynth_tree(tree)
            validation = validate_route(target, route)
        except Exception as exc:
            rejected.append(type(exc).__name__)
            continue
        if not route.steps:
            rejected.append("empty_route")
            continue
        if validation.route_score <= 0:
            rejected.append(",".join(validation.penalties) or "route_score_zero")
            continue
        route.starting_materials = validation.starting_materials
        route.intermediates = validation.intermediates
        if validation.route_score > best_score or (validation.route_score == best_score and best_route is not None and len(route.steps) < len(best_route.steps)):
            best_route = route
            best_score = validation.route_score
    if best_route is None:
        detail = "; ".join(rejected[:5]) or "no route trees parsed"
        raise ToolError(f"AiZynthFinder route trees contained no route passing hard validation: {detail}")
    return best_route


def _route_from_aizynth_tree(tree: Any) -> Route:
    if isinstance(tree, str):
        tree = json.loads(tree)
    if not isinstance(tree, dict):
        raise ToolError("AiZynthFinder route tree must be a JSON object")
    steps_with_depth: list[tuple[int, str]] = []
    starting_materials: list[str] = []

    def walk_molecule(node: dict[str, Any], depth: int) -> str:
        product = canonicalize_smiles(_node_smiles(node))
        reaction_children = [child for child in _node_children(node) if _node_is_reaction(child)]
        if not reaction_children:
            if product not in starting_materials:
                starting_materials.append(product)
            return product
        reaction = reaction_children[0]
        reactants = [
            walk_molecule(child, depth + 1)
            for child in _node_children(reaction)
            if isinstance(child, dict) and not _node_is_reaction(child)
        ]
        if not reactants:
            raise ToolError("AiZynthFinder reaction node has no molecule reactants")
        step = canonicalize_reaction_step(f"{'.'.join(reactants)}>>{product}")
        steps_with_depth.append((depth, step))
        return product

    walk_molecule(tree, 0)
    steps = [step for _, step in sorted(steps_with_depth, key=lambda item: item[0], reverse=True)]
    return Route(
        steps=steps,
        starting_materials=starting_materials,
        source="aizynthfinder",
        confidence="high",
        solved=True,
    )


def _node_smiles(node: dict[str, Any]) -> str:
    value = node.get("smiles") or node.get("smiles_str")
    if not isinstance(value, str) or not value.strip():
        raise ToolError("AiZynthFinder tree node missing SMILES")
    return value.strip()


def _node_children(node: dict[str, Any]) -> list[Any]:
    children = node.get("children", [])
    if not isinstance(children, list):
        return []
    return children


def _node_is_reaction(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = str(node.get("type", "")).lower()
    return node_type == "reaction" or ">>" in str(node.get("smiles", ""))


def find_executable(*names: str) -> str | None:
    env_bin = Path(sys.executable).resolve().parent
    for name in names:
        path = shutil.which(name)
        if path:
            return path
        env_path = env_bin / name
        if env_path.exists() and env_path.is_file():
            return str(env_path)
    return None


def require_executable(*names: str) -> str:
    path = find_executable(*names)
    if not path:
        raise ToolError(f"required executable not found: {' or '.join(names)}")
    return path


def _sbdd_command_available() -> bool:
    command = os.getenv("AI4S_SBDD_GENERATOR_CMD")
    if not command:
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return bool(parts and find_executable(parts[0]) is not None)


def has_python_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def _configured_path_exists(env_name: str) -> bool:
    value = os.getenv(env_name)
    return bool(value and Path(value).exists())


def safe_name(text: str, limit: int = 48) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return (cleaned.strip("._-") or "item")[:limit]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def run_command(command: list[str], timeout: int, env: dict[str, str] | None = None) -> CommandResult:
    start = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    return CommandResult(proc.returncode, proc.stdout, proc.stderr, time.monotonic() - start)


def _must_run(command: list[str], timeout: int) -> CommandResult:
    result = run_command(command, timeout=timeout)
    if result.returncode != 0:
        raise ToolError(f"command failed: {' '.join(command)}\n{result.stderr or result.stdout}")
    return result


def _parse_vina_energy(text: str) -> float:
    for line in text.splitlines():
        stripped = line.strip()
        if re_match := __import__("re").match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", stripped):
            return float(re_match.group(1))
    raise ToolError("Vina output did not contain rank-1 affinity")
