from __future__ import annotations

import os


def run_seed_from_env(default: int = 0) -> int:
    value = os.getenv("CHEM_EVOLVE_RUN_SEED")
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def offset_seed(base_seed: int, offset: int) -> int:
    return (int(base_seed) + int(offset)) % 2_147_483_647
