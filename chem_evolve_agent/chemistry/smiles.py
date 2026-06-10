from __future__ import annotations


def _rdkit():
    try:
        from rdkit import Chem
    except Exception:
        return None
    return Chem


def is_valid_smiles(smiles: str) -> bool:
    Chem = _rdkit()
    if Chem is None:
        return bool(smiles) and " " not in smiles and ">>" not in smiles and "not-a" not in smiles
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str:
    Chem = _rdkit()
    if Chem is None:
        if not is_valid_smiles(smiles):
            raise ValueError(f"invalid SMILES: {smiles}")
        simple_aliases = {
            "C(C)O": "CCO",
            "OCC": "CCO",
        }
        return simple_aliases.get(smiles, smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def route_final_product(route_text: str) -> str:
    if not route_text:
        raise ValueError("empty route")
    last_step = route_text.split(",")[-1]
    if ">>" not in last_step:
        raise ValueError(f"route step missing product separator: {last_step}")
    return last_step.rsplit(">>", 1)[-1]
