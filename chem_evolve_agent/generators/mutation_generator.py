from __future__ import annotations

import random

from chem_evolve_agent.generators.base import GenerationContext, MoleculeGenerator


class MutationGenerator(MoleculeGenerator):
    name = "mutation_generator"

    def __init__(self, seeds: list[str] | None = None, seed: int = 29):
        self._seeds = seeds or [
            "c1ccccc1",
            "c1ccncc1",
            "CC(=O)Nc1ccccc1",
            "NC(=O)c1ccccc1",
        ]
        self._random = random.Random(seed)
        self._templates = [
            "Cc1ccccc1",
            "COc1ccccc1",
            "Clc1ccccc1",
            "Fc1ccccc1",
            "Nc1ccccc1",
            "CC(=O)Nc1ccccc1",
            "CCOc1ccccc1",
            "Cc1ccncc1",
            "COc1ccncc1",
            "Clc1ccncc1",
            "Fc1ccncc1",
            "Nc1ccncc1",
            "CC(=O)Nc1ccncc1",
            "CCOc1ccncc1",
            "CNc1ccccc1",
            "CNc1ccncc1",
        ]

    def generate(self, context: GenerationContext, limit: int) -> list[str]:
        out: list[str] = []
        for _ in range(limit):
            if self._random.random() < 0.3:
                out.append(self._random.choice(self._seeds))
            else:
                out.append(self._random.choice(self._templates))
        return out
