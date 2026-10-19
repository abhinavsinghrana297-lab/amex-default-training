# Decision Log — AmEx Credit Default Prediction

Each entry covers what we decided, why, what else we could have done, and what it costs us.
Dead ends are logged too, because "what didn't work" is a common interview question.

---

## D-001 · Restart the project on Kaggle Notebooks
- **Decision:** Move the work from my laptop to Kaggle Notebooks (about 30 GB RAM and free GPU).
- **Why:** The first attempt was too slow locally. Checking features one by one by hand on 16 GB of data didn't scale.
- **Alternatives:** Stay local with chunked reading; use Colab; rent a cloud VM.
- **Trade-off:** Kaggle can't run FastAPI or Docker. So training happens on Kaggle, and serving and CI happen locally and on GitHub. The code lives in `src/` so the same files work in both places.

## D-002 · Convert the raw CSV to Parquet once, with smaller dtypes
- **Decision:** Stream the 16 GB CSV through Polars (`scan_csv` → `sink_parquet`), store floats as float32 and categoricals as small types, and never read the CSV again.
- **Why:** Parquet is columnar and compressed, so we can read only the columns we need. float32 roughly halves memory compared with float64, and loses no precision that matters here because the data is already noisy.
- **Alternatives:** Pandas with `chunksize`; the community pre-converted Parquet dataset by Raddar.
- **Trade-off:** It takes one slow conversion run. In return, every later step reads only a few GB.

## D-003 · Treat "time" as aggregation over each customer's statement history, not forecasting
- **Decision:** Each customer has up to 13 monthly statements, but there is one label per customer. We turn each customer's history into one row of summary features: last, mean, std, min, max, first, last − mean, last − first, and statement count.
- **Why:** The task is "will this customer default in the next 18 months?", not "predict next month's value". Summary features capture both the current state (last) and the trend (last − first).
- **Alternatives:** Use only the last statement (simple, but loses the trend). Sequence models such as GRU or Transformer (stronger, but heavy and hard to deploy). Kept as a possible stretch goal.
- **Trade-off:** The feature count grows to around 1,000+, so we need feature screening.

## D-004 · Univariate ROC-AUC is used for screening, not as the final feature selection
- **Decision:** Compute a per-feature AUC at the customer level, rank features by |AUC − 0.5|, and also compute the AUC of each feature's "is missing" flag.
- **Why:** It's fast, works without a threshold, isn't distorted by imbalance, and captures any monotonic relationship.
- **Caveats:** It's univariate, so it misses interactions and U-shaped effects. Tree models can use those. So we drop only clearly useless features. The strict cut is used only for the Logistic Regression baseline.
- **Leakage rule:** Selection is done on training folds only, never on the held-out set.

## D-005 · Hold out a customer-level test split
- **Decision:** After aggregation (one row per customer), create a stratified 80/20 split: 20% of customers are locked away as the final test set, and 5-fold stratified CV runs on the other 80%.
- **Why:** Kaggle's test labels are hidden. Splitting by customer means no customer's statements appear in both train and test.
- **Known limitation:** This split is "in-time". The real Kaggle test set came from later months, so our score may be slightly optimistic.

## D-006 · Class imbalance framing (be precise in interviews)
- **Fact:** The training set has about 26% defaulters only because AmEx **downsampled the non-defaulters to 5%**. The real population is heavily imbalanced, and the AMEX metric gives each negative a weight of 20 to undo the downsampling.
- **Decision:** Report imbalance this way, compare `scale_pos_weight` and class weights against no reweighting, and always evaluate with the true AMEX metric.

---

# Phase 0–1 · Setup, metric, CSV → Parquet

## D-007 · Write our own AMEX metric, and test it against the official one
- **Decision:** A vectorised NumPy version in `src/amex/evaluation/metric.py`, with a test that compares it to AmEx's own pandas code on 5 random datasets (must match to 1e-9).
- **Why:** The official version sorts three times and uses `.apply`, which is slow. We'll call the metric every boosting round for early stopping, so it must be fast. The test proves "fast" didn't change the answer.
- **Alternatives:** Copy the official code as-is (slow); optimise only for AUC (wrong target — AUC ignores the top-4% part of the score).
- **Interview line:** "The metric is half normalized Gini, half recall in the top 4% — it rewards both overall ranking and getting the riskiest customers right."

## D-008 · Convert only `train_data.csv`, not the 33 GB test file
- **Decision:** Skip the test file.
- **Why:** Its labels are hidden, so it can't score our model. Our held-out set comes from train (D-005).
- **Alternatives:** Convert it for a Kaggle leaderboard submission or adversarial validation (checking if train and test look different). The same function works on it if needed later.

## D-009 · Declare every column's type up front
- **Decision:** Pass a full schema to `scan_csv`: numerics read straight as Float32; ID, date and two text categories as strings; then convert categories to Int8 codes.
- **Why:** Without a schema, Polars guesses from the first rows. A column that is empty early on can be guessed as text, and every float is read as Float64 (twice the memory).
- **Detail:** Unknown category codes become null instead of crashing. This also protects the API later from unexpected input.

## D-010 · Shorten `customer_ID` from 64 hex characters to a 60-bit integer
- **Decision:** Keep the last 15 hex characters and store them as Int64.
- **Why:** ~5.5M rows × ~64 bytes of text vs 8 bytes. Joins and group-bys on integers are also much faster than on strings.
- **Risk & check:** Two customers could share the same 60 bits. With ~459k customers the chance is about 1 in 10 million, and `check_id_encoding` verifies uniqueness on the real labels file before converting.
- **Alternatives:** Keep the string; use Polars Categorical (still needs a dictionary of all strings); hash the full ID (same idea, but harder to check by eye).

## D-011 · Stream with `scan_csv` → `sink_parquet` (zstd)
- **Decision:** Never load the CSV into RAM. Polars processes it in batches and writes compressed Parquet.
- **Why:** Peak memory stays far below the 16 GB file size. zstd gives small files and fast reads.
- **Alternatives:** pandas `read_csv(chunksize=…)` and appending (more code, slower, easy to get dtypes inconsistent between chunks); Raddar's pre-made Parquet (fast, but not our own pipeline).
- **Not done (yet):** Raddar also found AmEx added small random noise to the floats and reversed it. We keep the floats as given; tree models are mostly unaffected. Possible later experiment.

## D-012 · Test on tiny synthetic data, and on two Polars versions
- **Decision:** `tests/conftest.py` builds a fake dataset with the same quirks (hex IDs, text categories, nulls, 1–13 statements per customer). Tests passed on Polars 1.0 and 2.0.
- **Why:** CI can't download 16 GB, and Kaggle's preinstalled Polars version may differ from my laptop's.

### Numbers to fill in after running notebook 01 on Kaggle
| Measure | Value |
|---|---|
| CSV size on disk | ___ GB |
| Parquet size on disk | ___ GB |
| DataFrame in RAM (Float32) | ___ GB |
| Conversion time | ___ min |
| Statements / customers | ___ / ___ |
