"""Check extracted official Task 1 checkpoints."""
from __future__ import annotations
import sys
from pathlib import Path

extracted_dir = Path("checkpoints/extracted")
official_checkpoints = [
    extracted_dir / "1D_Burgers_Sols_Nu0.001_FNO.pt",
    extracted_dir / "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt",
]

print(f"Extracted dir exists: {extracted_dir.exists()}")

# List extracted files
for f in sorted(extracted_dir.glob("**/*")):
    if f.is_file():
        print(f"  Found: {f}")

missing = [path for path in official_checkpoints if not path.exists()]
if missing:
    print("\nMissing official Task 1 checkpoints:")
    for path in missing:
        print(f"  {path}")
    sys.exit(1)

print("\nCheck complete. Ready to run official checkpoint ensemble.")
