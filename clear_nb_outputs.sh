#!/bin/bash
set -e
cd /home/itield7/TeraCyte_assignment
python3 <<'PY'
import json
from pathlib import Path
p = Path("analysis.ipynb")
nb = json.loads(p.read_text())
cleared = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code" and cell.get("outputs"):
        cell["outputs"] = []
        cell["execution_count"] = None
        cleared += 1
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Cleared outputs in {cleared} code cells")
PY
