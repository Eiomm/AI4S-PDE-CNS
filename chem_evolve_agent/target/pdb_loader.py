from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class AtomRecord(BaseModel):
    atom_name: str
    residue_name: str
    chain_id: str
    residue_id: int
    x: float
    y: float
    z: float


class PdbTarget(BaseModel):
    target_id: str
    path: Path
    atom_count: int
    has_hetatm: bool
    residues: List[str] = Field(default_factory=list)
    atoms: List[AtomRecord] = Field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        count = float(len(self.atoms))
        return (
            sum(atom.x for atom in self.atoms) / count,
            sum(atom.y for atom in self.atoms) / count,
            sum(atom.z for atom in self.atoms) / count,
        )

    @property
    def bounds(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        if not self.atoms:
            return None
        return (
            (
                min(atom.x for atom in self.atoms),
                min(atom.y for atom in self.atoms),
                min(atom.z for atom in self.atoms),
            ),
            (
                max(atom.x for atom in self.atoms),
                max(atom.y for atom in self.atoms),
                max(atom.z for atom in self.atoms),
            ),
        )


def load_pdb_target(path: Path) -> PdbTarget:
    text = path.read_text(errors="ignore") if path.exists() else ""
    atoms: List[AtomRecord] = []
    residues_seen = set()
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = _parse_atom_line(line)
        if atom is None:
            continue
        atoms.append(atom)
        residues_seen.add(f"{atom.chain_id}:{atom.residue_name}{atom.residue_id}")
    return PdbTarget(
        target_id=path.stem,
        path=path,
        atom_count=len(atoms),
        has_hetatm=any(line.startswith("HETATM") for line in text.splitlines()),
        residues=sorted(residues_seen),
        atoms=atoms,
    )


def _parse_atom_line(line: str) -> Optional[AtomRecord]:
    try:
        return AtomRecord(
            atom_name=line[12:16].strip(),
            residue_name=line[17:20].strip(),
            chain_id=line[21:22].strip() or "_",
            residue_id=int(line[22:26]),
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
        )
    except (ValueError, IndexError):
        return None
