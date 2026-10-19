import polars as pl
import pytest

from amex.data.convert import (
    check_id_encoding,
    convert_labels,
    convert_statements,
    summarize,
)
from amex.data.load import feature_columns, scan_labels, scan_statements


@pytest.fixture
def converted(cfg, fake_csvs):
    s_csv, l_csv, statements, labels = fake_csvs
    out = cfg["paths"]["processed_dir"]
    convert_statements(s_csv, f"{out}/train.parquet", cfg)
    convert_labels(l_csv, f"{out}/train_labels.parquet", cfg)
    return statements, labels


def test_dtypes(cfg, converted):
    schema = scan_statements(cfg).collect_schema()
    assert schema["customer_ID"] == pl.Int64
    assert schema["S_2"] == pl.Date
    for c in ["D_63", "D_64", "B_30", "D_68"]:
        assert schema[c] == pl.Int8, c
    for c in ["P_2", "D_39", "B_1", "R_1", "S_3"]:
        assert schema[c] == pl.Float32, c


def test_no_rows_or_customers_lost(cfg, converted):
    statements, labels = converted
    s = summarize(f"{cfg['paths']['processed_dir']}/train.parquet", cfg)
    assert s["rows"] == statements.height
    assert s["customers"] == labels.height


def test_nulls_preserved(cfg, converted):
    statements, _ = converted
    out = scan_statements(cfg).collect()
    for c in ["P_2", "B_30", "D_64"]:
        assert out[c].null_count() == statements[c].null_count(), c


def test_string_categories_mapped(cfg, converted):
    statements, _ = converted
    out = scan_statements(cfg, ["D_63"]).collect()["D_63"].to_list()
    mapping = cfg["data"]["string_category_maps"]["D_63"]
    assert out == [mapping[v] for v in statements["D_63"].to_list()]


def test_labels_join_back_to_statements(cfg, converted):
    _, labels = converted
    joined = (
        scan_statements(cfg, ["customer_ID"])
        .unique()
        .join(scan_labels(cfg), on="customer_ID", how="inner")
        .collect()
    )
    assert joined.height == labels.height


def test_id_encoding_unique(cfg, fake_csvs):
    check_id_encoding(fake_csvs[1], cfg)  # raises if two IDs collide


def test_feature_columns_excludes_id_and_date(cfg, converted):
    cols = feature_columns(cfg)
    assert "customer_ID" not in cols and "S_2" not in cols
    assert len(cols) == 9
