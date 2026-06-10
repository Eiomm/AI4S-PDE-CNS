#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_evolve_agent.tools.registry import list_tool_specs


def main() -> None:
    print("AI4S tool availability")
    print("=" * 88)
    for spec in list_tool_specs():
        status = "OK" if spec.available else "MISSING"
        programs = ", ".join(spec.required_programs)
        print(f"{status:<8} {spec.name:<32} {programs}")
        print(f"         {spec.purpose}")
        if spec.note:
            print(f"         备注：{spec.note}")


if __name__ == "__main__":
    main()
