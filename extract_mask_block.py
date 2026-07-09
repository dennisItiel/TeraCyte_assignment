import json
from pathlib import Path
nb = json.loads(Path("analysis.ipynb").read_text())
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    chunks = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            t = out.get("text", "")
            chunks.append(t if isinstance(t, str) else "".join(t))
    text = "".join(chunks)
    if "inside_px == expected_px" in text:
        print(text)
