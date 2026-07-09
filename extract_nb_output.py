import json
from pathlib import Path
nb = json.loads(Path("analysis.ipynb").read_text())
errors = []
validation_lines = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        if ot == "error":
            errors.append((i, out.get("ename"), out.get("evalue")))
        text = ""
        if ot == "stream":
            t = out.get("text", "")
            text = t if isinstance(t, str) else "".join(t)
        elif ot in ("execute_result", "display_data"):
            data = out.get("data", {})
            t = data.get("text/plain", "")
            text = t if isinstance(t, str) else "".join(t)
        for line in text.splitlines():
            if "inside_px" in line or "inside_not_bg" in line or "Outside" in line or "outside" in line.lower() and "circle" in line.lower():
                validation_lines.append(line)
            if "inside_px == expected_px" in line or "inside_not_bg == inside_px" in line:
                validation_lines.append(">>>" + line)
print("=== MASK VALIDATION LINES ===")
for line in validation_lines:
    print(line)
print("=== ERRORS ===")
if errors:
    for e in errors:
        print(e)
else:
    print("None")
print("=== CODE CELLS ===")
code_cells = [c for c in nb["cells"] if c.get("cell_type")=="code"]
executed = sum(1 for c in code_cells if c.get("execution_count") is not None)
print(f"{executed}/{len(code_cells)} code cells have execution_count")
