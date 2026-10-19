"""Shared test fixtures: a tiny fake version of the AmEx data.

CI can't download 16 GB, so tests run on a few hundred rows that have the same
shape and quirks as the real file: hex IDs, date strings, text categories,
float categories, missing values, and several statements per customer.
"""

import secrets

import numpy as np
import polars as pl
import pytest

from amex.config import load_config

NUMERIC = ["P_2", "D_39", "B_1", "R_1", "S_3"]
FLOAT_CATS = ["B_30", "D_68"]


def make_fake_statements(n_customers: int = 40, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = [secrets.token_hex(32) for _ in range(n_customers)]
    rows = []
    for cid in ids:
        n_statements = int(rng.integers(1, 14))  # 1..13 statements, like the real data
        for m in range(n_statements):
            row = {"customer_ID": cid, "S_2": f"2017-{m % 12 + 1:02d}-15"}
            for c in NUMERIC:
                row[c] = None if rng.random() < 0.2 else float(rng.normal())
            row["B_30"] = None if rng.random() < 0.1 else float(rng.integers(0, 3))
            row["D_68"] = float(rng.integers(0, 7))
            row["D_63"] = str(rng.choice(["CO", "CR", "CL", "XZ", "XM", "XL"]))
            row["D_64"] = None if rng.random() < 0.1 else str(rng.choice(["O", "U", "R", "-1"]))
            rows.append(row)
    statements = pl.DataFrame(rows)
    labels = pl.DataFrame({"customer_ID": ids, "target": rng.integers(0, 2, n_customers).tolist()})
    return statements, labels


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"]["processed_dir"] = str(tmp_path / "processed")
    return c


@pytest.fixture
def fake_csvs(tmp_path):
    statements, labels = make_fake_statements()
    s_path, l_path = tmp_path / "train_data.csv", tmp_path / "train_labels.csv"
    statements.write_csv(s_path)
    labels.write_csv(l_path)
    return s_path, l_path, statements, labels
