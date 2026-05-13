"""Check extracted checkpoints and run FNO ensemble inference on test set."""
from __future__ import annotations
import os
import tarfile
import sys
from pathlib import Path

# Check if checkpoints are extracted
extracted_dir = Path("checkpoints/extracted")
tar_path = Path("checkpoints/burgers_FNO.tar")

print(f"Extracted dir exists: {extracted_dir.exists()}")
print(f"Tar file exists: {tar_path.exists()}")

if not extracted_dir.exists():
    if tar_path.exists():
        print("Extracting burgers_FNO.tar...")
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extracted_dir)
        print("Extraction done.")
    else:
        print("ERROR: burgers_FNO.tar not found!")
        sys.exit(1)

# List extracted files
for f in sorted(extracted_dir.glob("**/*")):
    if f.is_file():
        print(f"  Found: {f}")

print("\nCheck complete. Ready to run ensemble.")
