# High-Frequency Volatility Forecasting

This project explores short-horizon S&P 500 volatility forecasting using GARCH and GARCH-X benchmarks alongside option-data-augmented GRU models.

## Project structure

- `Data/` — input and processed parquet datasets used by the notebooks
- `Notebooks/` — research and model notebooks
- `backend/` — reusable utility functions for metrics and GARCH-X recursion
- `project_paths.py` — portable data-path helper for local and GitHub-safe notebook execution
- `requirements.txt` — Python dependencies

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start Jupyter:
   `jupyter lab` or `jupyter notebook`
4. Open the notebooks in `Notebooks/`.

## Data assumptions

The notebooks expect to find these files in the repository:

- `Data/^SP500.Last.txt`
- `Data/train_features_5m_clean.parquet`
- `Data/processed_sets/*.parquet`

If you are running from a fresh clone, ensure those files are present before executing the notebooks. If the data is too large for a normal Git repository, store it in Git LFS or an external dataset bucket and keep the same folder structure.

## Notes for GitHub uploads

- The notebook cells use project-relative paths instead of machine-specific Windows paths.
- Large generated outputs such as Keras tuner folders and `.npz` files are ignored.
- The repository includes a lightweight backend module so the notebooks can work from a clean clone.

## Missing files and caveats

This repo is intentionally set up to work from a fresh checkout, but it still depends on the actual data files being present in `Data/`. If your local copy is missing any of those parquet or text files, the notebooks will fail until they are restored.
