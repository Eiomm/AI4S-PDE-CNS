from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationContext:
    target_id: str
    pocket_summary: str
    round_index: int
    run_seed: int = 0


class MoleculeGenerator:
    name = "base"

    def generate(self, context: GenerationContext, limit: int) -> list[str]:
        raise NotImplementedError
