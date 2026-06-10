from __future__ import annotations

import random

from chem_evolve_agent.generators.base import GenerationContext, MoleculeGenerator


class FragmentCatalogGenerator(MoleculeGenerator):
    name = "fragment_catalog_generator"

    def __init__(self, seed: int = 43):
        self._random = random.Random(seed)
        self._catalog = [
            "CCOc1ccccc1",
            "COc1ccccc1",
            "Clc1ccccc1",
            "Fc1ccccc1",
            "Cc1ccccc1",
            "Nc1ccccc1",
            "NC(=O)c1ccccc1",
            "CC(=O)Nc1ccccc1",
            "CCOc1ccncc1",
            "COc1ccncc1",
            "Clc1ccncc1",
            "Fc1ccncc1",
            "Cc1ccncc1",
            "Nc1ccncc1",
            "NC(=O)c1ccncc1",
            "CC(=O)Nc1ccncc1",
            "CNc1ccccc1",
            "CNc1ccncc1",
            "CCN1CCCCC1",
            "CN1CCCCC1",
            "CCN(CC)CC",
            "CCN(CC)CCO",
            "CCOC(=O)c1ccccc1",
            "COC(=O)c1ccccc1",
            "CC(C)Oc1ccccc1",
            "CC(C)Oc1ccncc1",
            "COc1ccc(NC(C)=O)cc1",
            "Cc1ccc(NC(C)=O)cc1",
            "O=C(Nc1ccccc1)c1ccccc1",
            "O=C(Nc1ccncc1)c1ccccc1",
            "c1ccc2[nH]ccc2c1",
            "COc1ccc2[nH]ccc2c1",
            "Cc1ccc2[nH]ccc2c1",
            "CN(C)c1ccccc1",
            "CN(C)c1ccncc1",
        ]

    def generate(self, context: GenerationContext, limit: int) -> list[str]:
        catalog = list(self._catalog)
        self._random.shuffle(catalog)
        return catalog[:limit]
