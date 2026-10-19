# AmEx Credit Default Prediction

End-to-end credit default prediction on the [American Express – Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) Kaggle dataset (16 GB of monthly card statements): a memory-efficient Polars pipeline, a benchmark of Logistic Regression vs XGBoost vs LightGBM scored with the competition's AMEX metric, and a FastAPI inference service packaged with Docker and tested with GitHub Actions.

> 🚧 Work in progress. Done so far: project setup, AMEX metric, CSV → Parquet conversion.

## The problem

Each customer has up to 13 monthly statements with ~190 anonymised features (D = delinquency, S = spend, P = payment, B = balance, R = risk). The task is to predict whether the customer defaults within 18 months after their latest statement.

**Metric:** `M = 0.5 × (normalized weighted Gini + share of defaulters caught in the top 4%)`, with non-defaulters weighted ×20 because AmEx kept only 5% of them in the data.

## Results

| Step | Result |
|---|---|
| CSV → Parquet | _TBD_ GB → _TBD_ GB on disk |
| Logistic Regression | _TBD_ |
| XGBoost | _TBD_ |
| LightGBM | _TBD_ |

## Project layout

```
configs/config.yaml        paths and settings (no hard-coded paths in code)
src/amex/data/             CSV -> Parquet conversion, lazy loading
src/amex/evaluation/       AMEX metric (checked against the official version)
src/amex/utils/            memory and timing helpers
notebooks/                 thin Kaggle notebooks that call src/
tests/                     pytest suite on small synthetic data
docs/decision_log.md       every key decision, alternative and trade-off
```

## Running

**On Kaggle (training):** open `notebooks/01_convert.ipynb`, add the competition data, turn Internet on, set `REPO_URL`, and run all.

**Locally (tests):**
```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Design decisions

See [`docs/decision_log.md`](docs/decision_log.md).
