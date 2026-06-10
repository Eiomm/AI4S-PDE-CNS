from __future__ import annotations

from typing import List, Optional

from chem_evolve_agent.generators.base import GenerationContext, MoleculeGenerator
from chem_evolve_agent.llm import LiteLlmClient
from chem_evolve_agent.chemistry.smiles import canonicalize_smiles, is_valid_smiles


class LlmGenerator(MoleculeGenerator):
    name = "llm_generator"

    def __init__(self, client: Optional[LiteLlmClient] = None):
        self.client = client or LiteLlmClient()

    def generate(self, context: GenerationContext, limit: int) -> List[str]:
        if not self.client.available:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You propose small-molecule SMILES for CNS-oriented protein targets. "
                    "Return only a JSON array of valid, synthetically plausible SMILES strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target id: {context.target_id}\n"
                    f"Pocket summary: {context.pocket_summary}\n"
                    f"Round: {context.round_index}\n"
                    f"Exploration seed: {context.run_seed}\n"
                    f"Return at most {limit} diverse molecules. Avoid salts and routes."
                ),
            },
        ]
        try:
            payload = self.client.complete_json(messages)
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        molecules: List[str] = []
        for item in payload:
            if not isinstance(item, str) or not is_valid_smiles(item):
                continue
            try:
                canonical = canonicalize_smiles(item)
            except ValueError:
                continue
            if canonical not in molecules:
                molecules.append(canonical)
            if len(molecules) >= limit:
                break
        return molecules
