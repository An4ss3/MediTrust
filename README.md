# MediTrust

**A Trustworthy Medical AI Pipeline Integrating Federated Learning, Confidence Calibration, Adversarial Robustness, Explainability, and Edge Deployment.**

Academic research prototype, course *Intelligence Artificielle Avancée*, ENSA Fès (3IACN), 2026/2027. Author: Anasse Harki.

> ⚠️ Research prototype only. Not a medical device, not clinically validated, not for diagnostic use.

## Status

Phase 1 (environment and data pipeline). No model has been trained yet. Progress is tracked in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Documentation (French)

| File | Content |
|---|---|
| [`docs/CAHIER_DES_CHARGES.md`](docs/CAHIER_DES_CHARGES.md) | Requirements, architecture, decisions |
| [`docs/PROTOCOLE_EXPERIMENTAL.md`](docs/PROTOCOLE_EXPERIMENTAL.md) | Experiments E1–E7, metrics, test-set discipline |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases and checklists |
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | Current progress |

## Layout

```
docs/        specification documents
src/         reusable code (data.py so far)
notebooks/   thin Colab notebooks calling src/
tests/       pytest tests
results/     small versioned results (partitions, metrics, figures)
```

Datasets, checkpoints and ONNX files are never committed. They live in `data/` locally or on Google Drive.

## Quick start

Local (Python ≥ 3.10):

```bash
pip install numpy matplotlib pytest          # torch optional locally
python -c "from src.data import download; download('data')"   # ~1.1 GB, MD5-checked
pytest -q
```

Google Colab: open [`notebooks/01_data.ipynb`](notebooks/01_data.ipynb) ("Open in Colab" from GitHub) and run all cells. It clones this repository, downloads the data to the VM (optional cache on Google Drive) and runs the integrity checks. CPU runtime is enough for Phase 1; approximate run time 5–10 min, mostly the download.

## Data

PathMNIST+ 64×64 from MedMNIST v3 (Zenodo [10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652)), official splits 89,996 / 10,004 / 7,180, nine tissue classes. Derived from NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al., Zenodo [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)). License CC BY 4.0.

Please cite: Yang et al., *MedMNIST v2*, Scientific Data 10, 41 (2023), and Kather et al., *PLOS Medicine* (2019).
