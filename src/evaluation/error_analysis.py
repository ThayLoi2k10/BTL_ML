"""Error analysis and model blind-spot diagnosis for Week 4.

Author: Danh (Person 3 - Error Analysis)
Responsibilities:
1. Receive the unified test predictions from Person 1 (``metrics.py``) and
   extract every misclassified test row (y_true != y_pred), plus the rows with
   the largest raw residual |y_true - y_prob| (the regression-style view).
2. Map each processed test row back to its original row in ``data/raw/bank.csv``
   (before cleaning, clipping, encoding and scaling) so the real attribute
   values can be inspected.
3. Profile False Positives / False Negatives: rare categories, rows close to the
   decision boundary, rows whose features were altered by imputation
   ('unknown' -> 'missing') or by IQR clipping.
4. Head-to-head comparison between model families (Tree vs Linear, ...):
   find rows one family gets right and the other gets wrong, with an exact
   McNemar test.
5. Select 5-10 representative errors and attach a rule-based technical
   diagnosis for each one, to be refined in report Section 5.2.

The split is reproduced exactly as in ``notebooks/04_features_pipeline.ipynb``
(``split_data`` with test_size=0.2, stratify=True, random_state=42).  The row
order of ``X_test.parquet`` / ``test_predictions.parquet`` is the order of the
raw ``X_test`` after ``reset_index(drop=True)``, so positional alignment plus a
label check is sufficient to recover the raw index.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import OutlierClipper  # noqa: E402
from src.encoding_scaling import TARGET_COLUMN, split_data  # noqa: E402
from src.pipeline import LEAKAGE_COLUMNS, load_correct_dataset  # noqa: E402

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.parquet"
OUTPUT_DIR = REPORTS_DIR / "error_analysis"

POS_LABEL = "yes"
NEG_LABEL = "no"

MODEL_KEYS = ("linear", "tree", "nonlinear")
MODEL_NAMES = {
    "dummy": "Dummy Baseline",
    "linear": "Logistic Regression",
    "tree": "Random Forest",
    "nonlinear": "SVC RBF",
}

NUMERIC_COLUMNS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
# Columns whose 'unknown' value is replaced by 'missing' in MissingHandler.
IMPUTED_COLUMNS = ["job", "education", "contact", "poutcome"]
# Columns clipped by OutlierClipper (duration is dropped before clipping).
CLIPPED_COLUMNS = ["age", "balance", "campaign", "previous", "pdays"]

# |p - 0.5| below this value -> "near the decision boundary".
BOUNDARY_MARGIN = 0.10
# |p - 0.5| above this value -> "confident error".
CONFIDENT_MARGIN = 0.30
# A category whose share in the training set is below this value is "rare".
RARE_THRESHOLD = 0.05
# Number of training neighbours used to measure local class overlap.
KNN_NEIGHBORS = 15


# ---------------------------------------------------------------------------
# 1. Loading and raw-index mapping
# ---------------------------------------------------------------------------
def load_predictions(path: str | Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the unified prediction table produced by Person 1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found at {path}. Run `python -m src.evaluation.metrics` first."
        )
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_raw_splits(
    data_path: str | Path = RAW_DATA_PATH,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce the Week-2 split on the raw CSV and validate it.

    Returns raw ``X_train, X_test, y_train, y_test`` whose index is the row
    position in ``bank.csv``.  Raises if the reproduced test labels do not match
    ``y_test.parquet`` row by row.
    """
    data = load_correct_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(data, target_column=TARGET_COLUMN)

    y_test_path = Path(processed_dir) / "y_test.parquet"
    if y_test_path.exists():
        y_saved = pd.read_parquet(y_test_path).iloc[:, 0]
        if len(y_saved) != len(y_test) or not (
            y_saved.to_numpy() == y_test.to_numpy()
        ).all():
            raise ValueError(
                "Reproduced raw test split does not match data/processed/y_test.parquet."
            )
    return X_train, X_test, y_train, y_test


def compute_clip_bounds(X_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Recompute the IQR clipping bounds that the pipeline learned on X_train."""
    clipper = OutlierClipper().fit(X_train.drop(columns=list(LEAKAGE_COLUMNS)))
    return {col: tuple(map(float, b)) for col, b in clipper.bounds_.items()}


def compute_train_frequencies(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Share of each category in the training set (used to flag rare groups)."""
    return {
        col: X_train[col].value_counts(normalize=True).to_dict()
        for col in CATEGORICAL_COLUMNS
    }


def compute_knn_yes_share(
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    n_neighbors: int = KNN_NEIGHBORS,
) -> np.ndarray:
    """Share of 'yes' among the k nearest *training* rows of each test row.

    Distances are computed in the processed 32-feature space the models
    actually see.  A FN whose neighbours are mostly 'no' (or a FP whose
    neighbours are mostly 'yes') lies in a region where the classes overlap:
    no model trained on these features can separate it reliably.
    """
    from sklearn.neighbors import NearestNeighbors

    processed_dir = Path(processed_dir)
    X_train = pd.read_parquet(processed_dir / "X_train.parquet").to_numpy(dtype=float)
    X_test = pd.read_parquet(processed_dir / "X_test.parquet").to_numpy(dtype=float)
    y_train = pd.read_parquet(processed_dir / "y_train.parquet").iloc[:, 0]
    y_train = (y_train == POS_LABEL).to_numpy(dtype=float)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    _, idx = nn.kneighbors(X_test)
    return y_train[idx].mean(axis=1)


def _outcome(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Label each row as TP / TN / FP / FN with 'yes' as the positive class."""
    t = y_true == POS_LABEL
    p = y_pred == POS_LABEL
    return pd.Series(
        np.select([t & p, ~t & ~p, ~t & p], ["TP", "TN", "FP"], default="FN"),
        index=y_true.index,
    )


def build_analysis_frame(
    predictions: pd.DataFrame | None = None,
    model_keys: tuple[str, ...] = MODEL_KEYS,
) -> pd.DataFrame:
    """Join raw test attributes with every model's prediction and diagnostics.

    Output columns
    --------------
    raw_index                  : row position in data/raw/bank.csv
    <raw features>             : original, unscaled values (``duration`` is kept
                                 for post-hoc diagnosis only; models never saw it)
    y_true, y_true_binary
    y_pred_<m>, y_prob_<m>     : from Person 1
    correct_<m>, outcome_<m>   : bool / TP-TN-FP-FN
    margin_<m>                 : |p - 0.5|, distance to the 0.5 threshold
    residual_<m>               : |y_true_binary - y_prob|
    near_boundary_<m>          : margin < BOUNDARY_MARGIN
    imputed_cols / n_imputed   : columns whose 'unknown' became 'missing'
    clipped_cols / n_clipped   : columns altered by IQR clipping
    rare_cols / n_rare         : categories with train share < RARE_THRESHOLD
    never_contacted            : pdays == -1 (no previous campaign)
    knn_yes_share              : share of 'yes' among the 15 nearest train rows
    """
    if predictions is None:
        predictions = load_predictions()

    X_train, X_test, _, y_test = load_raw_splits()
    if len(predictions) != len(X_test):
        raise ValueError("Prediction table and raw test split have different lengths.")
    if not (predictions["y_true"].to_numpy() == y_test.to_numpy()).all():
        raise ValueError("Prediction labels are not aligned with the raw test split.")

    frame = X_test.reset_index().rename(columns={"index": "raw_index"})
    frame = pd.concat([frame, predictions.reset_index(drop=True)], axis=1)

    for m in model_keys:
        prob = frame[f"y_prob_{m}"]
        frame[f"correct_{m}"] = frame[f"y_pred_{m}"] == frame["y_true"]
        frame[f"outcome_{m}"] = _outcome(frame["y_true"], frame[f"y_pred_{m}"])
        frame[f"margin_{m}"] = (prob - 0.5).abs()
        frame[f"residual_{m}"] = (frame["y_true_binary"] - prob).abs()
        frame[f"near_boundary_{m}"] = frame[f"margin_{m}"] < BOUNDARY_MARGIN

    # Imputation noise: 'unknown' is merged into a single 'missing' level.
    imputed = pd.DataFrame(
        {col: frame[col].astype(str).str.lower() == "unknown" for col in IMPUTED_COLUMNS}
    )
    frame["n_imputed"] = imputed.sum(axis=1)
    frame["imputed_cols"] = imputed.apply(
        lambda r: ",".join(c for c in IMPUTED_COLUMNS if r[c]), axis=1
    )
    # poutcome is 'unknown' for ~75% of rows (never contacted); keep a flag that
    # ignores it so the signal is not drowned by this structural missingness.
    frame["imputed_informative"] = imputed[["job", "education", "contact"]].any(axis=1)

    # Clipping noise: value outside the train IQR bounds (pdays=-1 is a sentinel).
    bounds = compute_clip_bounds(X_train)
    clipped = {}
    for col in CLIPPED_COLUMNS:
        lower, upper = bounds[col]
        values = frame[col]
        mask = (values < lower) | (values > upper)
        if col == "pdays":
            mask &= values >= 0
        clipped[col] = mask
    clipped = pd.DataFrame(clipped)
    frame["n_clipped"] = clipped.sum(axis=1)
    frame["clipped_cols"] = clipped.apply(
        lambda r: ",".join(c for c in CLIPPED_COLUMNS if r[c]), axis=1
    )

    # Rare groups in the training distribution.
    freqs = compute_train_frequencies(X_train)
    rare = pd.DataFrame(
        {
            col: frame[col].map(freqs[col]).fillna(0.0) < RARE_THRESHOLD
            for col in CATEGORICAL_COLUMNS
        }
    )
    frame["n_rare"] = rare.sum(axis=1)
    frame["rare_cols"] = rare.apply(
        lambda r: ",".join(f"{c}={frame.at[r.name, c]}" for c in CATEGORICAL_COLUMNS if r[c]),
        axis=1,
    )
    frame["never_contacted"] = frame["pdays"] == -1
    frame["knn_yes_share"] = compute_knn_yes_share()

    frame.attrs["clip_bounds"] = bounds
    frame.attrs["train_frequencies"] = freqs
    return frame


# ---------------------------------------------------------------------------
# 2. Extraction of wrong predictions / largest residuals
# ---------------------------------------------------------------------------
RAW_VIEW_COLUMNS = ["raw_index", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, "y_true"]


def extract_misclassified(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """All test rows where y_true != y_pred, most confident errors first."""
    cols = [
        *RAW_VIEW_COLUMNS,
        f"y_pred_{model_key}",
        f"y_prob_{model_key}",
        f"outcome_{model_key}",
        f"margin_{model_key}",
        "imputed_cols",
        "clipped_cols",
        "rare_cols",
        "knn_yes_share",
    ]
    wrong = frame.loc[~frame[f"correct_{model_key}"], cols]
    return wrong.sort_values(
        [f"margin_{model_key}", "raw_index"], ascending=[False, True]
    ).reset_index(drop=True)


def largest_absolute_errors(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    top_k: int = 20,
    index: pd.Index | np.ndarray | None = None,
) -> pd.DataFrame:
    """Generic top-k |y_true - y_score| extractor.

    For regression ``y_score`` is the predicted value; for classification it is
    the positive-class probability and ``y_true`` the 0/1 label.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    index = np.arange(len(y_true)) if index is None else np.asarray(index)
    table = pd.DataFrame(
        {
            "index": index,
            "y_true": y_true,
            "y_score": y_score,
            "error": y_score - y_true,
            "abs_error": np.abs(y_score - y_true),
        }
    )
    return table.sort_values(["abs_error", "index"], ascending=[False, True]).head(top_k)


def largest_residuals(frame: pd.DataFrame, model_key: str, top_k: int = 20) -> pd.DataFrame:
    """Top-k raw residuals of a model, joined back to raw attributes."""
    top = largest_absolute_errors(
        frame["y_true_binary"], frame[f"y_prob_{model_key}"], top_k, frame["raw_index"]
    )
    top = top.rename(
        columns={
            "index": "raw_index",
            "y_true": "y_true_binary",
            "y_score": f"y_prob_{model_key}",
        }
    ).reset_index(drop=True)
    raw = frame.set_index("raw_index")[RAW_VIEW_COLUMNS[1:]]
    return top.join(raw, on="raw_index")


# ---------------------------------------------------------------------------
# 3. FP / FN profiling
# ---------------------------------------------------------------------------
def summarize_outcomes(frame: pd.DataFrame, model_keys: tuple[str, ...] = MODEL_KEYS) -> pd.DataFrame:
    """Confusion counts and error-confidence statistics per model."""
    records = []
    for m in model_keys:
        outcome = frame[f"outcome_{m}"]
        wrong = ~frame[f"correct_{m}"]
        margin = frame.loc[wrong, f"margin_{m}"]
        records.append(
            {
                "model": MODEL_NAMES[m],
                "TP": int((outcome == "TP").sum()),
                "TN": int((outcome == "TN").sum()),
                "FP": int((outcome == "FP").sum()),
                "FN": int((outcome == "FN").sum()),
                "n_errors": int(wrong.sum()),
                "error_rate": float(wrong.mean()),
                "FN_share_of_errors": float((outcome == "FN").sum() / wrong.sum()),
                "errors_near_boundary_share": float((margin < BOUNDARY_MARGIN).mean()),
                "errors_confident_share": float((margin >= CONFIDENT_MARGIN).mean()),
                "median_error_margin": float(margin.median()),
            }
        )
    return pd.DataFrame(records)


def compare_error_groups(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Median raw numeric values and flag rates for TP / TN / FP / FN groups.

    Reading guide: FN should be compared with TP (same true class 'yes'),
    FP with TN (same true class 'no').
    """
    outcome = frame[f"outcome_{model_key}"]
    rows = {}
    for group in ("TP", "FN", "TN", "FP"):
        sub = frame[outcome == group]
        stats: dict[str, Any] = {"n": len(sub)}
        for col in NUMERIC_COLUMNS:
            stats[f"median_{col}"] = sub[col].median()
        stats["mean_prob"] = sub[f"y_prob_{model_key}"].mean()
        stats["share_never_contacted"] = sub["never_contacted"].mean()
        stats["share_poutcome_success"] = (sub["poutcome"] == "success").mean()
        stats["share_contact_unknown"] = (sub["contact"] == "unknown").mean()
        stats["share_month_may"] = (sub["month"] == "may").mean()
        stats["share_housing_yes"] = (sub["housing"] == "yes").mean()
        stats["share_any_loan"] = ((sub["housing"] == "yes") | (sub["loan"] == "yes")).mean()
        stats["share_imputed_informative"] = sub["imputed_informative"].mean()
        stats["share_clipped"] = (sub["n_clipped"] > 0).mean()
        stats["share_rare"] = (sub["n_rare"] > 0).mean()
        stats["share_near_boundary"] = sub[f"near_boundary_{model_key}"].mean()
        stats["mean_knn_yes_share"] = sub["knn_yes_share"].mean()
        # Neighbourhood majority agrees with the (wrong or right) prediction.
        predicted_yes = group in ("TP", "FP")
        knn_majority_yes = sub["knn_yes_share"] > 0.5
        stats["share_knn_agrees_with_prediction"] = (
            knn_majority_yes if predicted_yes else ~knn_majority_yes
        ).mean()
        rows[group] = stats
    return pd.DataFrame(rows)


def segment_error_rates(
    frame: pd.DataFrame,
    model_key: str,
    column: str,
    min_count: int = 1,
) -> pd.DataFrame:
    """Error, FP and FN rate per category of ``column`` (with train share)."""
    freqs = frame.attrs.get("train_frequencies", {}).get(column)
    grouped = frame.groupby(column, observed=True)
    table = pd.DataFrame(
        {
            "n_test": grouped.size(),
            "positive_rate": grouped["y_true_binary"].mean(),
            "error_rate": grouped[f"correct_{model_key}"].apply(lambda s: 1 - s.mean()),
            "fp_count": grouped[f"outcome_{model_key}"].apply(lambda s: (s == "FP").sum()),
            "fn_count": grouped[f"outcome_{model_key}"].apply(lambda s: (s == "FN").sum()),
            "mean_prob": grouped[f"y_prob_{model_key}"].mean(),
        }
    )
    if freqs is not None:
        table.insert(1, "train_share", [freqs.get(k, 0.0) for k in table.index])
        table["is_rare"] = table["train_share"] < RARE_THRESHOLD
    table = table[table["n_test"] >= min_count]
    return table.sort_values("error_rate", ascending=False)


def flag_error_rates(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Error rate with vs. without each data-quality / difficulty flag."""
    wrong = ~frame[f"correct_{model_key}"]
    flags = {
        "near_boundary (|p-0.5|<0.1)": frame[f"near_boundary_{model_key}"],
        "knn_overlap (neighbours mostly opposite label)": (
            (frame["knn_yes_share"] > 0.5) != (frame["y_true"] == POS_LABEL)
        ),
        "rare_category (train share<5%)": frame["n_rare"] > 0,
        "imputed job/education/contact": frame["imputed_informative"],
        "contact=unknown": frame["contact"] == "unknown",
        "clipped (any IQR clipping)": frame["n_clipped"] > 0,
        "balance clipped": frame["clipped_cols"].str.contains("balance"),
        "campaign clipped": frame["clipped_cols"].str.contains("campaign"),
        "never_contacted (pdays=-1)": frame["never_contacted"],
        "poutcome=success": frame["poutcome"] == "success",
    }
    records = []
    for name, mask in flags.items():
        mask = mask.astype(bool)
        records.append(
            {
                "flag": name,
                "n_with_flag": int(mask.sum()),
                "error_rate_with_flag": float(wrong[mask].mean()) if mask.any() else np.nan,
                "error_rate_without_flag": float(wrong[~mask].mean()) if (~mask).any() else np.nan,
                "share_of_all_errors": float((wrong & mask).sum() / wrong.sum()),
            }
        )
    table = pd.DataFrame(records)
    table["lift"] = table["error_rate_with_flag"] / table["error_rate_without_flag"]
    return table


def post_hoc_duration_check(frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Median call ``duration`` per outcome group.

    ``duration`` is removed from training because it is only known after the
    call.  It is used here purely as a diagnostic: if FN rows have long calls
    (like TP) and FP rows have short calls (like TN), the errors are driven by
    information that does not exist at prediction time (irreducible error for a
    pre-call model) rather than by a model defect.
    """
    return (
        frame.groupby(f"outcome_{model_key}")["duration"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .loc[["TP", "FN", "TN", "FP"], ["count", "25%", "50%", "75%", "mean"]]
    )


# ---------------------------------------------------------------------------
# 4. Head-to-head comparison between model families
# ---------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * p))


def head_to_head(frame: pd.DataFrame, model_a: str, model_b: str) -> dict[str, Any]:
    """Contingency of correctness between two models plus McNemar test."""
    a = frame[f"correct_{model_a}"]
    b = frame[f"correct_{model_b}"]
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    return {
        "model_a": MODEL_NAMES[model_a],
        "model_b": MODEL_NAMES[model_b],
        "both_correct": int((a & b).sum()),
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": int((~a & ~b).sum()),
        "mcnemar_p_value": mcnemar_exact(only_a, only_b),
    }


def disagreement_cases(frame: pd.DataFrame, winner: str, loser: str) -> pd.DataFrame:
    """Rows ``winner`` gets right and ``loser`` gets wrong (loser's most confident first)."""
    mask = frame[f"correct_{winner}"] & ~frame[f"correct_{loser}"]
    cols = [
        *RAW_VIEW_COLUMNS,
        f"y_prob_{winner}",
        f"y_prob_{loser}",
        f"outcome_{loser}",
        f"margin_{loser}",
    ]
    return frame.loc[mask, cols].sort_values(
        [f"margin_{loser}", "raw_index"], ascending=[False, True]
    ).reset_index(drop=True)


def disagreement_profile(frame: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Compare raw attributes of 'only A correct' vs 'only B correct' vs 'both correct'."""
    a = frame[f"correct_{model_a}"]
    b = frame[f"correct_{model_b}"]
    groups = {
        f"only_{model_a}_correct": a & ~b,
        f"only_{model_b}_correct": ~a & b,
        "both_correct": a & b,
        "both_wrong": ~a & ~b,
    }
    rows = {}
    for name, mask in groups.items():
        sub = frame[mask]
        stats: dict[str, Any] = {
            "n": len(sub),
            "positive_rate": sub["y_true_binary"].mean(),
            "median_age": sub["age"].median(),
            "median_balance": sub["balance"].median(),
            "share_balance_clipped": sub["clipped_cols"].str.contains("balance").mean(),
            "share_age_60_plus": (sub["age"] >= 60).mean(),
            "share_age_under_30": (sub["age"] < 30).mean(),
            "median_day": sub["day"].median(),
            "median_campaign": sub["campaign"].median(),
            "share_never_contacted": sub["never_contacted"].mean(),
            "share_poutcome_success": (sub["poutcome"] == "success").mean(),
            "share_contact_unknown": (sub["contact"] == "unknown").mean(),
            "share_month_may": (sub["month"] == "may").mean(),
            "share_rare_month": sub["month"].isin(["mar", "sep", "oct", "dec"]).mean(),
            f"mean_prob_{model_a}": sub[f"y_prob_{model_a}"].mean(),
            f"mean_prob_{model_b}": sub[f"y_prob_{model_b}"].mean(),
            f"mean_margin_{model_a}": sub[f"margin_{model_a}"].mean(),
            f"mean_margin_{model_b}": sub[f"margin_{model_b}"].mean(),
        }
        rows[name] = stats
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Representative errors with technical diagnosis
# ---------------------------------------------------------------------------
def diagnose_error(row: pd.Series, model_key: str, frame: pd.DataFrame) -> str:
    """Rule-based technical explanation (Vietnamese) for one wrong prediction."""
    outcome = row[f"outcome_{model_key}"]
    prob = row[f"y_prob_{model_key}"]
    margin = row[f"margin_{model_key}"]
    reasons: list[str] = []

    if margin < BOUNDARY_MARGIN:
        reasons.append(f"Sát biên quyết định (p={prob:.2f}); chỉ cần dịch ngưỡng nhẹ là đổi nhãn")
    elif margin >= CONFIDENT_MARGIN:
        reasons.append(f"Lỗi tự tin cao (p={prob:.2f}): hồ sơ trông rất giống lớp đối lập")

    if "knn_yes_share" in row:
        share = row["knn_yes_share"]
        true_share = share if row["y_true"] == POS_LABEL else 1 - share
        if true_share < 0.5:
            n_opposite = round((1 - true_share) * KNN_NEIGHBORS)
            reasons.append(
                f"Chồng lấn lớp: {n_opposite}/{KNN_NEIGHBORS} láng giềng gần nhất trong Train "
                "mang nhãn ngược với nhãn thật"
            )

    yes_rows = frame["y_true"] == POS_LABEL
    duration_yes_median = frame.loc[yes_rows, "duration"].median()
    duration_no_median = frame.loc[~yes_rows, "duration"].median()

    if outcome == "FN":
        if row["never_contacted"]:
            reasons.append("Chưa từng được liên hệ (pdays=-1, poutcome=unknown): thiếu tín hiệu lịch sử dương")
        if row["month"] == "may":
            reasons.append("Tháng 5 (month=may) mang hệ số âm mạnh do chiến dịch đại trà tỉ lệ thành công thấp")
        if row["contact"] == "unknown":
            reasons.append("contact=unknown bị gộp thành 'missing' (hệ số âm) dù khách thực tế đăng ký")
        if row["housing"] == "yes" or row["loan"] == "yes":
            reasons.append("Đang có khoản vay (housing/loan=yes) kéo xác suất xuống")
        if row["campaign"] >= 4:
            reasons.append(f"Bị gọi nhiều lần (campaign={row['campaign']}) thường gắn với từ chối")
        if row["duration"] >= duration_yes_median:
            reasons.append(
                f"Hậu kiểm: cuộc gọi dài (duration={row['duration']}s ≥ trung vị nhóm yes "
                f"{duration_yes_median:.0f}s) - quyết định hình thành trong cuộc gọi, "
                "thông tin bị loại vì leakage"
            )
    else:  # FP
        if row["poutcome"] == "success":
            reasons.append("poutcome=success (đặc trưng mạnh nhất) lấn át các tín hiệu khác")
        if row["month"] in ("mar", "sep", "oct", "dec"):
            reasons.append(f"Tháng hiếm có tỉ lệ yes cao (month={row['month']}) đẩy xác suất lên")
        if row["age"] >= 60 or row["age"] < 25:
            reasons.append(f"Nhóm tuổi biên (age={row['age']}) có tỉ lệ yes cao trong tập Train")
        if row["housing"] == "no" and row["loan"] == "no" and row["balance"] > 1500:
            reasons.append(f"Không vay, số dư cao (balance={row['balance']}): hồ sơ tài chính 'đẹp'")
        if row["duration"] <= duration_no_median:
            reasons.append(
                f"Hậu kiểm: cuộc gọi ngắn (duration={row['duration']}s ≤ trung vị nhóm no "
                f"{duration_no_median:.0f}s) - khách từ chối nhanh, mô hình trước cuộc gọi không thể biết"
            )

    if row["clipped_cols"]:
        bounds = frame.attrs.get("clip_bounds", {})
        parts = []
        for col in row["clipped_cols"].split(","):
            lower, upper = bounds.get(col, (np.nan, np.nan))
            target = upper if row[col] > upper else lower
            parts.append(f"{col}={row[col]} bị clip về {target:g}")
        reasons.append("Nhiễu do cắt outlier: " + "; ".join(parts))
    if row["imputed_cols"]:
        reasons.append(f"Giá trị 'unknown' được điền thành 'missing' ở: {row['imputed_cols']}")
    if row["rare_cols"]:
        reasons.append(f"Thuộc nhóm hiếm trong Train (<5%): {row['rare_cols']}")

    return "; ".join(reasons) if reasons else "Không có đặc điểm nổi bật - hồ sơ trung tính"


def select_representative_errors(
    frame: pd.DataFrame,
    model_key: str = "tree",
    compare_with: str = "linear",
    n_samples: int = 10,
) -> pd.DataFrame:
    """Pick up to ``n_samples`` diverse, typical errors and diagnose each one.

    Selection slots (first match not already chosen, deterministic order):
    confident FN, confident FP, near-boundary FN, near-boundary FP,
    FN with clipped balance, FP with poutcome=success, FN with contact=unknown,
    error of ``model_key`` that ``compare_with`` gets right, the reverse case
    (``compare_with`` wrong, ``model_key`` right), and a row all models miss.
    """
    outcome = frame[f"outcome_{model_key}"]
    margin = frame[f"margin_{model_key}"]
    wrong = ~frame[f"correct_{model_key}"]
    others_wrong = np.logical_and.reduce(
        [~frame[f"correct_{m}"] for m in MODEL_KEYS if f"correct_{m}" in frame]
    )

    slots = [
        ("FN tự tin cao", outcome == "FN", margin, False),
        ("FP tự tin cao", outcome == "FP", margin, False),
        ("FN sát biên", (outcome == "FN") & frame[f"near_boundary_{model_key}"], margin, True),
        ("FP sát biên", (outcome == "FP") & frame[f"near_boundary_{model_key}"], margin, True),
        ("FN có balance bị clip", (outcome == "FN") & frame["clipped_cols"].str.contains("balance"), margin, False),
        ("FP do poutcome=success", (outcome == "FP") & (frame["poutcome"] == "success"), margin, False),
        ("FN có contact=unknown", (outcome == "FN") & (frame["contact"] == "unknown"), margin, False),
        (
            f"{MODEL_NAMES[model_key]} sai, {MODEL_NAMES[compare_with]} đúng",
            wrong & frame[f"correct_{compare_with}"],
            margin,
            False,
        ),
        (
            f"{MODEL_NAMES[compare_with]} sai, {MODEL_NAMES[model_key]} đúng",
            ~frame[f"correct_{compare_with}"] & frame[f"correct_{model_key}"],
            frame[f"margin_{compare_with}"],
            False,
        ),
        ("Cả 3 mô hình cùng sai", pd.Series(others_wrong, index=frame.index), margin, False),
    ]

    chosen: list[int] = []
    labels: list[str] = []
    for label, mask, score, ascending in slots:
        if len(chosen) >= n_samples:
            break
        candidates = frame.loc[mask & ~frame.index.isin(chosen)].copy()
        if candidates.empty:
            continue
        candidates["_score"] = score[candidates.index]
        candidates = candidates.sort_values(["_score", "raw_index"], ascending=[ascending, True])
        chosen.append(int(candidates.index[0]))
        labels.append(label)

    selected = frame.loc[chosen].copy()
    selected.insert(0, "case_type", labels)
    # Diagnose with the model that actually made the error in the reverse slot.
    diag_model = [
        compare_with if (not row[f"correct_{compare_with}"] and row[f"correct_{model_key}"]) else model_key
        for _, row in selected.iterrows()
    ]
    selected["diagnosed_model"] = [MODEL_NAMES[m] for m in diag_model]
    selected["diagnosis"] = [
        diagnose_error(row, m, frame) for (_, row), m in zip(selected.iterrows(), diag_model)
    ]
    keep = [
        "case_type",
        "raw_index",
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
        "y_true",
        *[f"y_prob_{m}" for m in MODEL_KEYS],
        "knn_yes_share",
        "diagnosed_model",
        "diagnosis",
    ]
    return selected[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. End-to-end runner
# ---------------------------------------------------------------------------
def run_error_analysis(
    output_dir: str | Path = OUTPUT_DIR,
    primary_model: str = "tree",
    compare_with: str = "linear",
) -> dict[str, pd.DataFrame]:
    """Run every analysis step and export CSV tables to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_analysis_frame()
    results: dict[str, pd.DataFrame] = {
        "outcome_summary": summarize_outcomes(frame),
        "misclassified_" + primary_model: extract_misclassified(frame, primary_model),
        "misclassified_" + compare_with: extract_misclassified(frame, compare_with),
        "largest_residuals_" + primary_model: largest_residuals(frame, primary_model, top_k=20),
        "group_profile_" + primary_model: compare_error_groups(frame, primary_model),
        "group_profile_" + compare_with: compare_error_groups(frame, compare_with),
        "flag_error_rates_" + primary_model: flag_error_rates(frame, primary_model),
        "duration_check_" + primary_model: post_hoc_duration_check(frame, primary_model),
        "representative_errors": select_representative_errors(frame, primary_model, compare_with),
    }
    for column in ("job", "education", "month", "poutcome", "contact"):
        results[f"segment_{column}_{primary_model}"] = segment_error_rates(frame, primary_model, column)

    pairs = [("tree", "linear"), ("nonlinear", "linear"), ("tree", "nonlinear")]
    results["head_to_head"] = pd.DataFrame([head_to_head(frame, a, b) for a, b in pairs])
    results["disagreement_profile_tree_vs_linear"] = disagreement_profile(frame, "tree", "linear")
    results["tree_right_linear_wrong"] = disagreement_cases(frame, "tree", "linear")
    results["linear_right_tree_wrong"] = disagreement_cases(frame, "linear", "tree")

    for name, table in results.items():
        index = not isinstance(table.index, pd.RangeIndex)
        table.to_csv(output_dir / f"{name}.csv", index=index, encoding="utf-8-sig")

    results["_frame"] = frame
    return results


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    out = run_error_analysis()

    print("--- OUTCOME SUMMARY ---")
    print(out["outcome_summary"].round(4).to_string(index=False))
    print("\n--- ERROR RATE BY FLAG (Random Forest) ---")
    print(out["flag_error_rates_tree"].round(4).to_string(index=False))
    print("\n--- HEAD TO HEAD ---")
    print(out["head_to_head"].to_string(index=False))
    print("\n--- REPRESENTATIVE ERRORS ---")
    for _, r in out["representative_errors"].iterrows():
        print(f"[{r['case_type']}] raw_index={r['raw_index']} y_true={r['y_true']}: {r['diagnosis']}")
    print(f"\nTables exported to {OUTPUT_DIR}")
