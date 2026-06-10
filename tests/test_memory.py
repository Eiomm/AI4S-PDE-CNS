from chem_evolve_agent.evolution.memory import MemoryBank


def test_memory_promotes_repeated_observation():
    bank = MemoryBank()
    promoted = bank.promote_observations(
        [
            {"rule": "pyridine core improved score", "context": {"branch": "a"}},
            {"rule": "pyridine core improved score", "context": {"branch": "b"}},
        ]
    )
    assert len(promoted) == 2
    assert bank.records[0].rule == "pyridine core improved score"


def test_memory_promotes_large_single_branch_improvement():
    bank = MemoryBank()
    promoted = bank.promote_observations(
        [{"rule": "amide tail helped route", "score_improvement": 0.12}]
    )
    assert len(promoted) == 1
    assert promoted[0].confidence > 0.5
