from chem_evolve_agent.chemistry.smiles import is_valid_smiles
from chem_evolve_agent.generators.base import GenerationContext
from chem_evolve_agent.generators.fragment_generator import FragmentCatalogGenerator
from chem_evolve_agent.generators.mutation_generator import MutationGenerator
from chem_evolve_agent.generators.seed_generator import SeedGenerator


def test_seed_generator_outputs_valid_smiles():
    context = GenerationContext(target_id="target", pocket_summary="none", round_index=0)
    assert all(is_valid_smiles(smiles) for smiles in SeedGenerator(seed=1).generate(context, 20))


def test_mutation_generator_outputs_valid_smiles():
    context = GenerationContext(target_id="target", pocket_summary="none", round_index=0)
    assert all(is_valid_smiles(smiles) for smiles in MutationGenerator(seed=1).generate(context, 20))


def test_fragment_catalog_generator_outputs_valid_smiles():
    context = GenerationContext(target_id="target", pocket_summary="none", round_index=0)
    generated = FragmentCatalogGenerator(seed=1).generate(context, 30)
    assert len(set(generated)) == len(generated)
    assert all(is_valid_smiles(smiles) for smiles in generated)
