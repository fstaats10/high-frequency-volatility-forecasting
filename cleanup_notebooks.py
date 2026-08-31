import json
from pathlib import Path

root = Path(__file__).resolve().parent

replacements = {
    "Notebooks/DNN.ipynb": [
        ("C:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\note\\test_forecasts.npz", "../note/test_forecasts.npz"),
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\34_features_newvol\\processed_sets\\removed_peak_month.parquet", "../Data/processed_sets/removed_peak_month.parquet"),
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\34_features_newvol\\processed_sets\\winsorized_1_99.parquet", "../Data/processed_sets/winsorized_1_99.parquet"),
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\34_features_newvol\\processed_sets\\regime_high_vol.parquet", "../Data/processed_sets/regime_high_vol.parquet"),
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\34_features_newvol\\processed_sets\\regime_low_vol.parquet", "../Data/processed_sets/regime_low_vol.parquet"),
    ],
    "Notebooks/GARCH.ipynb": [
        ("C:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\note\\garch_spx_test_forecasts.npz", "../note/garch_spx_test_forecasts.npz"),
    ],
    "Notebooks/stats.ipynb": [
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\clean_data_12\\train_features_5m_clean.parquet", "../Data/train_features_5m_clean.parquet"),
        ("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash\\34_features_newvol\\train_features_5m_clean.parquet", "../Data/train_features_5m_clean.parquet"),
    ],
}

for rel_path, reps in replacements.items():
    fp = root / rel_path
    text = fp.read_text(encoding="utf-8")
    original = text
    for old, new in reps:
        text = text.replace(old, new)
    if text != original:
        fp.write_text(text, encoding="utf-8")
        print(f"updated {rel_path}")

for old_name in ["Data/data3.ipynb", "Data/data4.ipynb"]:
    old_path = root / old_name
    if old_path.exists():
        old_path.unlink()
        print(f"removed {old_name}")

# Also normalize any remaining absolute path strings in file bodies.
for fp in sorted(root.rglob("*.ipynb")):
    text = fp.read_text(encoding="utf-8")
    if "C:\\Users\\Franswa" in text or "Option data dash" in text:
        text = text.replace("C:\\Users\\Franswa\\Documents\\TickStream\\Option data dash", "")
        text = text.replace("c:\\Users\\Franswa\\Documents\\TickStream\\Option data dash", "")
        fp.write_text(text, encoding="utf-8")
        print(f"stripped stale path from {fp.relative_to(root)}")

print("cleanup_notebooks_complete")
