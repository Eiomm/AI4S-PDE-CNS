from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles


def test_canonicalize_smiles_normalizes_valid_smiles():
    assert canonicalize_smiles("C(C)O") in {"CCO", "OCC"}


def test_invalid_smiles_returns_false():
    assert is_valid_smiles("not-a-smiles") is False
