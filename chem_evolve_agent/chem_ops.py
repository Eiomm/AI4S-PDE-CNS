from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from chem_evolve_agent.models import Route, Score


class RouteValidation(BaseModel):
    final_product: str | None = None
    final_product_matches: bool = False
    route_validity_score: float = 0.0
    starting_material_availability_score: float = 0.0
    step_penalty_score: float = 0.0
    convergence_score: float = 0.0
    balance_score: float = 0.0
    route_score: float = 0.0
    starting_materials: list[str] = Field(default_factory=list)
    intermediates: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)


def require_rdkit():
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("RDKit is required; no text-only SMILES fallback is allowed") from exc
    return Chem


def canonicalize_smiles(smiles: str) -> str:
    Chem = require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    try:
        canonicalize_smiles(smiles)
    except Exception:
        return False
    return True


def split_route_steps(route_text: str) -> list[str]:
    steps = [step.strip() for step in route_text.split(",") if step.strip()]
    if not steps:
        raise ValueError("empty route")
    return steps


def parse_reaction_step(step: str) -> tuple[list[str], list[str], list[str]]:
    if ">>" in step:
        reactants_text, products_text = step.split(">>", 1)
        agents_text = ""
    else:
        parts = step.split(">")
        if len(parts) != 3:
            raise ValueError(f"route step missing reaction separator: {step}")
        reactants_text, agents_text, products_text = parts
    reactants = _split_reaction_side(reactants_text)
    agents = _split_reaction_side(agents_text)
    products = _split_reaction_side(products_text)
    if not reactants:
        raise ValueError(f"route step missing reactant: {step}")
    if not products:
        raise ValueError(f"route step missing product: {step}")
    return reactants, agents, products


def route_final_product(route_text: str) -> str:
    _, _, products = parse_reaction_step(split_route_steps(route_text)[-1])
    return products[-1]


def canonicalize_reaction_step(step: str) -> str:
    reactants, agents, products = parse_reaction_step(step)
    lhs = ".".join(sorted(canonicalize_smiles(smiles) for smiles in reactants))
    rhs = ".".join(sorted(canonicalize_smiles(smiles) for smiles in products))
    if agents:
        middle = ".".join(sorted(canonicalize_smiles(smiles) for smiles in agents))
        return f"{lhs}>{middle}>{rhs}"
    return f"{lhs}>>{rhs}"


def element_counts(smiles: str) -> Counter[str]:
    Chem = require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


def reaction_element_imbalance(step: str) -> dict[str, int]:
    reactants, _, products = parse_reaction_step(step)
    reactant_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    for smiles in reactants:
        reactant_counts.update(element_counts(smiles))
    for smiles in products:
        product_counts.update(element_counts(smiles))
    return {
        element: count - reactant_counts.get(element, 0)
        for element, count in product_counts.items()
        if count > reactant_counts.get(element, 0)
    }


def validate_route(smiles: str, route: Route) -> RouteValidation:
    validation = RouteValidation()
    try:
        target = canonicalize_smiles(smiles)
        steps = [canonicalize_reaction_step(step) for step in route.steps]
        parsed = [parse_reaction_step(step) for step in steps]
        final_product = canonicalize_smiles(route_final_product(",".join(steps)))
    except Exception:
        validation.penalties.append("route_unparseable")
        return validation

    validation.final_product = final_product
    validation.final_product_matches = final_product == target
    validation.starting_materials = route.starting_materials or _starting_materials(parsed)
    validation.intermediates = route.intermediates or _intermediates(parsed)

    invalid = _invalid_route_molecules(parsed)
    if invalid:
        validation.penalties.append("route_invalid_molecule")
        return validation

    validation.route_validity_score = 1.0
    validation.balance_score = 0.0 if any(reaction_element_imbalance(step) for step in steps) else 1.0
    if validation.balance_score == 0.0:
        validation.penalties.append("route_element_imbalance")
    if not validation.final_product_matches:
        validation.penalties.append("route_product_mismatch")
    if _has_self_reaction(parsed):
        validation.penalties.append("route_self_reaction")

    validation.starting_material_availability_score = _starting_material_availability(validation.starting_materials, route)
    validation.step_penalty_score = clamp01(1.0 - max(0, len(steps) - 3) * 0.15)
    validation.convergence_score = 1.0 if len(validation.intermediates) <= len(steps) else 0.6
    validation.route_score = route_score(
        route_validity_score=validation.route_validity_score,
        starting_material_availability_score=validation.starting_material_availability_score,
        step_penalty_score=validation.step_penalty_score,
        convergence_score=validation.convergence_score,
        balance_score=validation.balance_score,
        final_product_matches=validation.final_product_matches,
        has_self_reaction="route_self_reaction" in validation.penalties,
    )
    return validation


def property_metrics(smiles: str) -> tuple[dict[str, float], list[str]]:
    Chem = require_rdkit()
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    metrics: dict[str, float] = {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "qed": float(QED.qed(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    }
    raw_sa = sascore_raw_from_smiles(canonicalize_smiles(smiles))
    if raw_sa is not None:
        metrics["sascore_raw"] = raw_sa
    warnings: list[str] = []
    if metrics["mw"] < 120 or metrics["mw"] > 650:
        warnings.append("mw_out_of_range")
    if metrics["logp"] < -1.5 or metrics["logp"] > 6.0:
        warnings.append("logp_out_of_range")
    if metrics["hbd"] > 5:
        warnings.append("hbd_high")
    if metrics["hba"] > 10:
        warnings.append("hba_high")
    if metrics["rotatable_bonds"] > 12:
        warnings.append("too_flexible")
    if abs(metrics["formal_charge"]) > 1:
        warnings.append("large_formal_charge")
    return metrics, warnings


def proxy_binding_score(metrics: dict[str, float], smiles: str) -> float:
    size_score = clamp01(float(metrics["heavy_atoms"]) / 35.0)
    simplicity_bonus = clamp01(1.0 - max(0, len(smiles) - 80) / 80.0)
    cns_logp_score = clamp01(1.0 - abs(float(metrics["logp"]) - 2.4) / 4.0)
    qed_score = clamp01(float(metrics["qed"]))
    prior_score = benchmark_prior_property_score(metrics)
    return round(0.25 * size_score + 0.20 * simplicity_bonus + 0.20 * qed_score + 0.15 * cns_logp_score + 0.20 * prior_score, 4)


def benchmark_prior_property_score(metrics: dict[str, float], prior_path: Path | None = None) -> float:
    prior = _load_benchmark_prior(prior_path)
    active_summary = prior.get("active_summary")
    if not isinstance(active_summary, dict):
        raise RuntimeError("benchmark prior missing active_summary")
    features = ["mw", "logp", "qed", "tpsa", "hbd", "hba", "rotatable_bonds", "heavy_atoms", "ring_count", "aromatic_rings", "sascore_raw"]
    scores = []
    for feature in features:
        if feature not in metrics or feature not in active_summary:
            continue
        item = active_summary[feature]
        if not isinstance(item, dict):
            continue
        scores.append(_prior_feature_score(float(metrics[feature]), item))
    if not scores:
        raise RuntimeError("benchmark prior has no usable descriptor overlap")
    return round(sum(scores) / len(scores), 4)


def binding_score_from_vina_energy(energy: float) -> float:
    return round(clamp01((-energy - 4.0) / 8.0), 4)


def sa_score(metrics: dict[str, float]) -> float:
    if "sascore_raw" not in metrics:
        raise RuntimeError("SA score is required for molecule scoring")
    raw = metrics["sascore_raw"]
    if raw > 4.0:
        return 0.0
    return round(clamp01((4.0 - raw) / 3.0), 4)


def molecule_score(binding_score: float, validity_score: float, sa_score_value: float) -> float:
    if clamp01(validity_score) <= 0.0:
        return 0.0
    return round(0.80 * clamp01(binding_score) + 0.10 * clamp01(validity_score) + 0.10 * clamp01(sa_score_value), 4)


def route_score(
    *,
    route_validity_score: float,
    starting_material_availability_score: float,
    step_penalty_score: float,
    convergence_score: float,
    balance_score: float,
    final_product_matches: bool,
    has_self_reaction: bool,
) -> float:
    if not final_product_matches or has_self_reaction:
        return 0.0
    if clamp01(route_validity_score) <= 0.0 or clamp01(balance_score) <= 0.0:
        return 0.0
    return round(
        0.55 * clamp01(route_validity_score)
        + 0.30 * clamp01(starting_material_availability_score)
        + 0.05 * clamp01(step_penalty_score)
        + 0.05 * clamp01(convergence_score)
        + 0.05 * clamp01(balance_score),
        4,
    )


def build_score(
    *,
    smiles: str,
    route: Route,
    binding_score_value: float,
    binding_source: str,
    docking_energy: float | None = None,
) -> Score:
    canonical = canonicalize_smiles(smiles)
    metrics, warnings = property_metrics(canonical)
    validity = 1.0
    sa_value = sa_score(metrics)
    prior_value = benchmark_prior_property_score(metrics)
    route_validation = validate_route(canonical, route)
    return Score(
        molecule_score=molecule_score(binding_score_value, validity, sa_value),
        route_score=route_validation.route_score,
        binding_score=binding_score_value,
        binding_source=binding_source,
        validity_score=validity,
        route_validity_score=route_validation.route_validity_score,
        starting_material_availability_score=route_validation.starting_material_availability_score,
        step_penalty_score=route_validation.step_penalty_score,
        convergence_score=route_validation.convergence_score,
        balance_score=route_validation.balance_score,
        docking_energy=docking_energy,
        qed=metrics.get("qed"),
        sa=sa_value,
        property_prior_score=prior_value,
        penalties=[*warnings, *route_validation.penalties],
    )


def generate_internal_smiles(limit: int, seed: int) -> list[str]:
    Chem = require_rdkit()

    base = [
        "CC(=O)Nc1ccccc1",
        "COc1ccccc1N",
        "Nc1ccc(Cl)cc1",
        "O=C(N)c1ccccc1Cl",
        "COc1ccc(N)cc1",
    ]
    substituents = ["F", "Cl", "C", "OC", "C(=O)N"]
    base = _rotate(base, seed)
    substituents = _rotate(substituents, seed // max(1, len(base)))
    out: list[str] = []
    for index, parent_smiles in enumerate(base):
        parent = Chem.MolFromSmiles(parent_smiles)
        if parent is None:
            continue
        sites = [
            atom.GetIdx()
            for atom in parent.GetAtoms()
            if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
        ]
        for site in sites:
            for sub in substituents:
                if len(out) >= limit:
                    return out
                candidate = _add_substituent(Chem, parent, site, sub)
                if candidate and candidate not in out:
                    out.append(candidate)
    return out[:limit]


def generate_evolved_smiles(parent_smiles: list[str], limit: int, seed: int) -> list[str]:
    Chem = require_rdkit()
    substituents = _rotate(["F", "Cl", "C", "OC", "C(=O)N"], seed)
    parents = _rotate([canonicalize_smiles(item) for item in parent_smiles], seed)
    out: list[str] = []
    for parent_smiles_item in parents:
        parent = Chem.MolFromSmiles(parent_smiles_item)
        if parent is None:
            continue
        sites = [
            atom.GetIdx()
            for atom in parent.GetAtoms()
            if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
        ]
        for site in _rotate(sites, seed):
            for sub in substituents:
                if len(out) >= limit:
                    return out
                candidate = _add_substituent(Chem, parent, site, sub)
                if candidate and candidate != parent_smiles_item and candidate not in out:
                    out.append(candidate)
    return out[:limit]


def _rotate(items: list[str], offset: int) -> list[str]:
    if not items:
        return items
    index = offset % len(items)
    return [*items[index:], *items[:index]]


@lru_cache(maxsize=4)
def _load_benchmark_prior(prior_path: Path | None = None) -> dict:
    path = prior_path or Path(__file__).resolve().parents[1] / "data/benchmarks/benchmark_prior.json"
    if not path.exists():
        raise RuntimeError(f"benchmark prior is required for proxy scoring: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _prior_feature_score(value: float, summary: dict[str, float]) -> float:
    q1 = float(summary["q1"])
    q3 = float(summary["q3"])
    median = float(summary["median"])
    if q1 <= value <= q3:
        return 1.0
    spread = max(q3 - q1, abs(median) * 0.25, 1.0)
    distance = q1 - value if value < q1 else value - q3
    return clamp01(1.0 - distance / spread)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def sascore_raw_from_smiles(smiles: str) -> float:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except Exception:
        raise RuntimeError("RDKit SA score dependencies are required")
    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return float(_calculate_sascore(Chem.MolToSmiles(mol, canonical=True)))


@lru_cache(maxsize=8192)
def _calculate_sascore(canonical_smiles: str) -> float:
    from rdkit import Chem
    from rdkit.Contrib.SA_Score import sascorer

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {canonical_smiles}")
    return float(sascorer.calculateScore(mol))


def _split_reaction_side(text: str) -> list[str]:
    return [item.strip() for item in text.split(".") if item.strip()]


def _starting_materials(parsed_steps: Iterable[tuple[list[str], list[str], list[str]]]) -> list[str]:
    produced: set[str] = set()
    starting: list[str] = []
    for reactants, _, products in parsed_steps:
        for reactant in reactants:
            canonical = canonicalize_smiles(reactant)
            if canonical not in produced and canonical not in starting:
                starting.append(canonical)
        produced.update(canonicalize_smiles(product) for product in products)
    return starting


def _intermediates(parsed_steps: Iterable[tuple[list[str], list[str], list[str]]]) -> list[str]:
    products: list[str] = []
    for _, _, step_products in parsed_steps:
        products.extend(canonicalize_smiles(product) for product in step_products)
    return products[:-1]


def _invalid_route_molecules(parsed_steps: Iterable[tuple[list[str], list[str], list[str]]]) -> list[str]:
    invalid: list[str] = []
    for reactants, agents, products in parsed_steps:
        for smiles in [*reactants, *agents, *products]:
            if not is_valid_smiles(smiles):
                invalid.append(smiles)
    return invalid


def _has_self_reaction(parsed_steps: Iterable[tuple[list[str], list[str], list[str]]]) -> bool:
    for reactants, _, products in parsed_steps:
        lhs = {canonicalize_smiles(reactant) for reactant in reactants}
        rhs = {canonicalize_smiles(product) for product in products}
        if lhs & rhs:
            return True
    return False


def _starting_material_availability(starting_materials: list[str], route: Route | None = None) -> float:
    if not starting_materials:
        return 0.0
    if route is not None and route.source == "aizynthfinder" and route.solved is True:
        return 1.0
    scores = []
    common_reagents = {"CC(=O)Cl", "N", "CCBr", "CI", "O", "CO", "CCO"}
    for material in starting_materials:
        canonical = canonicalize_smiles(material)
        metrics, warnings = property_metrics(canonical)
        if canonical in common_reagents:
            scores.append(1.0)
        elif metrics["heavy_atoms"] <= 16 and metrics["ring_count"] <= 2 and len(warnings) <= 1:
            scores.append(0.65)
        elif metrics["heavy_atoms"] <= 28 and len(warnings) <= 2:
            scores.append(0.35)
        else:
            scores.append(0.10)
    return round(sum(scores) / len(scores), 4)


def _add_substituent(Chem, mol, atom_idx: int, substituent: str) -> str | None:
    rw_mol = Chem.RWMol(mol)
    if substituent == "OC":
        oxygen_idx = rw_mol.AddAtom(Chem.Atom("O"))
        carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
        rw_mol.AddBond(atom_idx, oxygen_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(oxygen_idx, carbon_idx, Chem.BondType.SINGLE)
    elif substituent == "C(=O)N":
        carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
        oxygen_idx = rw_mol.AddAtom(Chem.Atom("O"))
        nitrogen_idx = rw_mol.AddAtom(Chem.Atom("N"))
        rw_mol.AddBond(atom_idx, carbon_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(carbon_idx, oxygen_idx, Chem.BondType.DOUBLE)
        rw_mol.AddBond(carbon_idx, nitrogen_idx, Chem.BondType.SINGLE)
    else:
        new_atom_idx = rw_mol.AddAtom(Chem.Atom(substituent))
        rw_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol, canonical=True)
    except Exception:
        return None
