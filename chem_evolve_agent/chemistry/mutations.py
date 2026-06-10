from __future__ import annotations

import random

from chem_evolve_agent.chemistry.smiles import _rdkit, canonicalize_smiles, is_valid_smiles


def simple_tail_mutations(smiles: str, seed: int = 0, limit: int = 4) -> list[str]:
    rng = random.Random(seed)
    tails = ["C", "OC", "F", "Cl", "N", "C(=O)N"]
    return [f"{rng.choice(tails)}{smiles}" for _ in range(limit)]


def rdkit_guided_mutations(smiles: str, seed: int = 0, limit: int = 8) -> list[str]:
    """Generate local, sanitized molecule variants around a parent SMILES.

    The edits are intentionally conservative: aromatic ring substituent additions,
    aromatic carbon-to-nitrogen swaps, and a small curated fallback neighborhood.
    Every result is RDKit-sanitized and canonicalized before it leaves this module.
    """

    Chem = _rdkit()
    if Chem is None:
        return _validated_unique(simple_tail_mutations(smiles, seed=seed, limit=limit * 2), limit=limit, parent=smiles)

    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        return []

    rng = random.Random(seed)
    candidates: list[str] = []

    # 先在芳香环上做保守取代：RWMol 加原子/加键后必须 Sanitize，
    # 避免回到字符串拼接导致大量非法 SMILES。
    aromatic_sites = _aromatic_carbon_h_sites(parent)
    rng.shuffle(aromatic_sites)
    for atom_idx in aromatic_sites:
        for substituent in rng.sample(_SUBSTITUENTS, k=len(_SUBSTITUENTS)):
            mutated = _add_substituent(Chem, parent, atom_idx, substituent)
            if mutated:
                candidates.append(mutated)
            if len(_validated_unique(candidates, limit=limit, parent=smiles)) >= limit:
                return _validated_unique(candidates, limit=limit, parent=smiles)

    # 再尝试芳香 C -> N 的轻量杂环化；这类变换可能改善 CNS-like 分子的相互作用模式。
    hetero_sites = _aromatic_carbon_h_sites(parent)
    rng.shuffle(hetero_sites)
    for atom_idx in hetero_sites[: min(4, len(hetero_sites))]:
        mutated = _aromatic_carbon_to_nitrogen(Chem, parent, atom_idx)
        if mutated:
            candidates.append(mutated)
        if len(_validated_unique(candidates, limit=limit, parent=smiles)) >= limit:
            return _validated_unique(candidates, limit=limit, parent=smiles)

    candidates.extend(_catalog_neighborhood(smiles))
    return _validated_unique(candidates, limit=limit, parent=smiles)


_SUBSTITUENTS = ("F", "Cl", "C", "N", "OC")


def _aromatic_carbon_h_sites(mol) -> list[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
        and atom.GetAtomicNum() == 6
        and atom.GetTotalNumHs() > 0
    ]


def _add_substituent(Chem, mol, atom_idx: int, substituent: str) -> str | None:
    rw_mol = Chem.RWMol(mol)
    if substituent == "OC":
        oxygen_idx = rw_mol.AddAtom(Chem.Atom("O"))
        carbon_idx = rw_mol.AddAtom(Chem.Atom("C"))
        rw_mol.AddBond(atom_idx, oxygen_idx, Chem.BondType.SINGLE)
        rw_mol.AddBond(oxygen_idx, carbon_idx, Chem.BondType.SINGLE)
    else:
        new_atom_idx = rw_mol.AddAtom(Chem.Atom(substituent))
        rw_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
    return _sanitize_to_smiles(Chem, rw_mol.GetMol())


def _aromatic_carbon_to_nitrogen(Chem, mol, atom_idx: int) -> str | None:
    rw_mol = Chem.RWMol(mol)
    atom = rw_mol.GetAtomWithIdx(atom_idx)
    atom.SetAtomicNum(7)
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(True)
    return _sanitize_to_smiles(Chem, rw_mol.GetMol())


def _sanitize_to_smiles(Chem, mol) -> str | None:
    try:
        # SanitizeMol 是合法化闸门；失败的中间结构不进入候选池。
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _catalog_neighborhood(smiles: str) -> list[str]:
    # A tiny safe neighborhood for common CNS-like aromatic parents. These are
    # still validated below; they only prevent total failure when direct edits
    # produce too few unique valid analogs.
    catalog = [
        "CCOc1ccccc1",
        "COc1ccccc1",
        "Fc1ccccc1",
        "Clc1ccccc1",
        "Cc1ccccc1",
        "CCOc1ccncc1",
        "COc1ccncc1",
        "Fc1ccncc1",
        "Clc1ccncc1",
        "Cc1ccncc1",
        "FCCOc1ccccc1",
        "NCCOc1ccccc1",
        "OCCCOc1ccccc1",
    ]
    return [item for item in catalog if item != smiles]


def _validated_unique(candidates: list[str], limit: int, parent: str) -> list[str]:
    out: list[str] = []
    try:
        canonical_parent = canonicalize_smiles(parent)
    except ValueError:
        canonical_parent = parent
    for candidate in candidates:
        # 最后一层统一过滤：合法、canonical、去 parent、去重复。
        if not candidate or not is_valid_smiles(candidate):
            continue
        try:
            canonical = canonicalize_smiles(candidate)
        except ValueError:
            continue
        if canonical == canonical_parent or canonical in out:
            continue
        out.append(canonical)
        if len(out) >= limit:
            break
    return out
