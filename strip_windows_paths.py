import json
from pathlib import Path

root = Path(__file__).resolve().parent
patterns = [
    ("C:\\Users\\Franswa\\Documents\\TickStream\\Option data dash", "../Data"),
    ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash", "../Data"),
    ("C:\\Users\\Franswa\\Documents\\TickStream\\options_tickstream\\high-frequency-volatility-forecasting", "."),
    ("c:\\Users\\Franswa\\Documents\\TickStream\\options_tickstream\\high-frequency-volatility-forecasting", "."),
]

for fp in sorted(root.rglob("*.ipynb")):
    try:
        nb = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"skipped invalid JSON: {fp} -> {exc}")
        continue

    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            new = value
            for old, rep in patterns:
                new = new.replace(old, rep)
            return new
        return value

    updated = walk(nb)
    if updated != nb:
        fp.write_text(json.dumps(updated, indent=1), encoding="utf-8")
        print(f"updated {fp.relative_to(root)}")

print("windows_path_stripping_complete")
