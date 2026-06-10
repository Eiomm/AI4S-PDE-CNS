from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from chem_evolve_agent.tools.base import find_executable, has_python_module
from chem_evolve_agent.chemistry.smiles import _rdkit


class ToolSpec(BaseModel):
    name: str
    purpose: str
    required_programs: list[str]
    available: bool
    note: str = ""


def list_tool_specs() -> list[ToolSpec]:
    aizynth_config = os.getenv("AIZYNTHFINDER_CONFIG")
    aizynth_config_ready = bool(aizynth_config and Path(aizynth_config).exists())
    return [
        ToolSpec(
            name="rdkit_property_tool",
            purpose="SMILES 合法性、理化性质、QED、Lipinski 风险检查",
            required_programs=["python:rdkit"],
            available=_rdkit() is not None,
        ),
        ToolSpec(
            name="fpocket_tool",
            purpose="从 PDB 靶点预测 binding pocket，并产出 pocket 描述文件",
            required_programs=["fpocket"],
            available=find_executable("fpocket") is not None,
        ),
        ToolSpec(
            name="obabel_meeko_prepare_tool",
            purpose="把 receptor/ligand 准备成 Vina 可读的 PDBQT",
            required_programs=["mk_prepare_ligand.py or obabel", "mk_prepare_receptor.py or obabel"],
            available=(
                find_executable("mk_prepare_ligand.py", "mk_prepare_ligand", "obabel", "babel") is not None
                and find_executable("mk_prepare_receptor.py", "mk_prepare_receptor", "obabel", "babel") is not None
            ),
        ),
        ToolSpec(
            name="vina_docking_tool",
            purpose="用 AutoDock Vina 对 ligand-receptor 进行 docking/score_only",
            required_programs=["vina command or python:vina"],
            available=find_executable("vina") is not None or has_python_module("vina"),
        ),
        ToolSpec(
            name="aizynthfinder_route_tool",
            purpose="调用 AiZynthFinder 做多步逆合成路线规划",
            required_programs=["aizynthcli", "AIZYNTHFINDER_CONFIG"],
            available=find_executable("aizynthcli", "aizynthfinder") is not None,
            note="" if aizynth_config_ready else "AIZYNTHFINDER_CONFIG 未配置；CLI 已安装，但真实逆合成会先跳过。",
        ),
    ]
