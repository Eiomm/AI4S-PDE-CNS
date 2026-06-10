from chem_evolve_agent.chemistry.mutations import rdkit_guided_mutations
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles


def test_rdkit_guided_mutations_are_valid_unique_and_not_parent():
    parent = canonicalize_smiles("FCCOc1ccccc1")
    generated = rdkit_guided_mutations(parent, seed=103, limit=8)

    assert len(generated) >= 4
    assert len(set(generated)) == len(generated)
    assert parent not in generated
    assert all(is_valid_smiles(smiles) for smiles in generated)
