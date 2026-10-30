from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
    root_mean_squared_error,
)

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR

# Default Model Metadata for Display
DEFAULT_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "dummy": {
        "name": "Dummy Baseline",
        "color": "#7f8c8d",  # Neutral Gray
        "linestyle": ":",
        "marker": "x",
    },
    "linear": {
        "name": "Logistic Regression (Tuned)",
        "color": "#2980b9",  # Deep Steel Blue
        "linestyle": "--",
        "marker": "^",
    },
    "tree": {
        "name": "Random Forest (Tuned)",
        "color": "#27ae60",  # Emerald Green (Champion)
        "linestyle": "-",
        "marker": "o",
    },
    "nonlinear": {
        "name": "SVC RBF (Tuned)",
        "color": "#e67e22",  # Amber / Orange
        "linestyle": "-.",
        "marker": "s",
    },
}


def setup_plot_style() -> None:
    """Thiết lập cấu hình giao diện đồ thị theo chuẩn xuất bản ấn phẩm."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "figure.autolayout": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_predictions_data(
    predictions_input: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Nạp dữ liệu DataFrame chứa nhãn thực tế và kết quả dự đoán của các mô hình."""
    if isinstance(predictions_input, pd.DataFrame):
        return predictions_input.copy()

    if predictions_input is None:
        parquet_path = REPORTS_DIR / "test_predictions.parquet"
        csv_path = REPORTS_DIR / "test_predictions.csv"
        if parquet_path.exists():
            predictions_input = parquet_path
        elif csv_path.exists():
            predictions_input = csv_path
        else:
            raise FileNotFoundError(
                f"Cannot find default predictions file in {REPORTS_DIR}. "
                "Ensure evaluation metrics (src.evaluation.metrics) have executed."
            )

    path = Path(predictions_input)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def plot_confusion_matrices(
    data: pd.DataFrame | dict[str, Any] | None = None,
    model_keys: Sequence[str] | None = None,
    pos_label: str = "yes",
    output_path: str | Path | None = FIGURES_DIR / "06_confusion_matrices.png",
    dpi: int = 300,
    save_vector: bool = False,
) -> plt.Figure:
    """Vẽ ma trận nhầm lẫn tuyệt đối và chuẩn hóa cho tất cả các mô hình."""
    setup_plot_style()
    df = load_predictions_data(data)

    if model_keys is None:
        # Detect available models in dataframe
        available_keys = [
            col.replace("y_pred_", "")
            for col in df.columns
            if col.startswith("y_pred_")
        ]
        # Preserve preferred order
        preferred_order = ["dummy", "linear", "tree", "nonlinear"]
        model_keys = [k for k in preferred_order if k in available_keys]
        if not model_keys:
            model_keys = available_keys

    num_models = len(model_keys)
    if num_models == 0:
        raise ValueError("No models found to construct confusion matrices.")

    labels = ["no", "yes"]
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_models,
        figsize=(4.2 * num_models, 8.5),
        squeeze=False,
    )

    y_true = df["y_true"].astype(str)

    for col_idx, key in enumerate(model_keys):
        model_name = DEFAULT_MODEL_CONFIGS.get(key, {}).get("name", key.capitalize())
        pred_col = f"y_pred_{key}"
        if pred_col not in df.columns:
            continue

        y_pred = df[pred_col].astype(str)
        cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
        with np.errstate(all="ignore"):
            cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        # ------------------ TOP ROW: RAW COUNTS ------------------ #
        ax_raw = axes[0, col_idx]
        annot_raw = np.empty_like(cm_raw, dtype=object)
        for i in range(2):
            for j in range(2):
                annot_raw[i, j] = f"{cell_names[i, j]}\n{cm_raw[i, j]:,}"

        sns.heatmap(
            cm_raw,
            annot=annot_raw,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax_raw,
            annot_kws={"fontsize": 11, "fontweight": "bold", "linespacing": 1.4},
            square=True,
            linewidths=1.0,
            linecolor="#f0f0f0",
        )
        ax_raw.set_title(
            f"{model_name}\n(Raw Counts)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax_raw.set_xticklabels(["Pred No", "Pred Yes"], fontsize=10)
        if col_idx == 0:
            ax_raw.set_yticklabels(["True No", "True Yes"], fontsize=10, va="center")
            ax_raw.set_ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
        else:
            ax_raw.set_yticklabels([])
            ax_raw.set_ylabel("")

        # ------------------ BOTTOM ROW: NORMALIZED ------------------ #
        ax_norm = axes[1, col_idx]
        annot_norm = np.empty_like(cm_norm, dtype=object)
        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100
                annot_norm[i, j] = f"{cell_names[i, j]}\n{pct:.1f}%"

        sns.heatmap(
            cm_norm,
            annot=annot_norm,
            fmt="",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            cbar=(col_idx == num_models - 1),
            cbar_kws={"label": "Normalized Recall / Rate"} if (col_idx == num_models - 1) else None,
            ax=ax_norm,
            annot_kws={"fontsize": 11, "fontweight": "bold", "linespacing": 1.4},
            square=True,
            linewidths=1.0,
            linecolor="#f0f0f0",
        )
        ax_norm.set_title(
            f"{model_name}\n(Row Normalized - Recall)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax_norm.set_xticklabels(["Pred No", "Pred Yes"], fontsize=10)
        ax_norm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
        if col_idx == 0:
            ax_norm.set_yticklabels(["True No", "True Yes"], fontsize=10, va="center")
            ax_norm.set_ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
        else:
            ax_norm.set_yticklabels([])
            ax_norm.set_ylabel("")

    fig.suptitle(
        "Diagnostic Performance Comparison: Side-by-Side Confusion Matrices\n"
        "(Top Row: Absolute Frequencies | Bottom Row: Row-Normalized Recall / Specificity)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        if save_vector:
            fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")

    return fig


def plot_roc_curves(
    data: pd.DataFrame | dict[str, Any] | None = None,
    model_keys: Sequence[str] | None = None,
    pos_label: str = "yes",
    output_path: str | Path | None = FIGURES_DIR / "07_roc_curves.png",
    dpi: int = 300,
    save_vector: bool = False,
) -> plt.Figure:
    """Vẽ đường cong ROC và tính diện tích AUC cho tất cả các mô hình trên cùng đồ thị."""
    setup_plot_style()
    df = load_predictions_data(data)

    if model_keys is None:
        available_keys = [
            col.replace("y_prob_", "")
            for col in df.columns
            if col.startswith("y_prob_")
        ]
        preferred_order = ["tree", "nonlinear", "linear", "dummy"]
        model_keys = [k for k in preferred_order if k in available_keys]
        if not model_keys:
            model_keys = available_keys

    y_true_binary = (
        df["y_true_binary"].to_numpy()
        if "y_true_binary" in df.columns
        else (df["y_true"] == pos_label).astype(int).to_numpy()
    )

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Reference diagonal: Random Guessing / No Skill
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#7f8c8d",
        linewidth=1.5,
        alpha=0.8,
        label="Chance Level (AUC = 0.5000)",
    )

    threshold_annotations: list[dict[str, Any]] = []

    for key in model_keys:
        prob_col = f"y_prob_{key}"
        pred_col = f"y_pred_{key}"
        if prob_col not in df.columns:
            continue

        y_prob = df[prob_col].to_numpy()
        cfg = DEFAULT_MODEL_CONFIGS.get(key, {})
        name = cfg.get("name", key.capitalize())
        color = cfg.get("color", "#2c3e50")
        linestyle = cfg.get("linestyle", "-")

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        lw = 2.5 if key == "tree" else 1.8
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=0.9,
        )

        # Highlight default operating point at tau = 0.5
        if pred_col in df.columns and key != "dummy":
            y_pred = df[pred_col].astype(str)
            # Operating TPR and FPR
            actual_pos = y_true_binary == 1
            actual_neg = y_true_binary == 0
            op_tpr = np.mean(y_pred[actual_pos] == pos_label)
            op_fpr = np.mean(y_pred[actual_neg] == pos_label)

            ax.scatter(
                op_fpr,
                op_tpr,
                color=color,
                edgecolor="#ffffff",
                s=90,
                zorder=5,
                linewidth=1.5,
            )

            # Locate Youden's J optimal cutoff: max(TPR - FPR)
            j_stat = tpr - fpr
            best_idx = np.argmax(j_stat)
            threshold_annotations.append({
                "name": name,
                "key": key,
                "auc": roc_auc,
                "op_fpr": op_fpr,
                "op_tpr": op_tpr,
                "best_thresh": thresholds[best_idx],
                "best_tpr": tpr[best_idx],
                "best_fpr": fpr[best_idx],
                "best_j": j_stat[best_idx],
            })

    # Add explanatory visual box for markers
    ax.scatter(
        [], [],
        color="#2c3e50",
        edgecolor="#ffffff",
        s=80,
        linewidth=1.5,
        label=r"Default Operating Point ($\tau = 0.50$)",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Receiver Operating Characteristic (ROC) Multi-Model Comparison\n"
        "Evaluation on Independent Test Set (N = 2,233)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        if save_vector:
            fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")

    return fig


def plot_pr_curves(
    data: pd.DataFrame | dict[str, Any] | None = None,
    model_keys: Sequence[str] | None = None,
    pos_label: str = "yes",
    output_path: str | Path | None = FIGURES_DIR / "08_pr_curves.png",
    dpi: int = 300,
    save_vector: bool = False,
) -> plt.Figure:
    """Vẽ đường cong Precision-Recall và tính điểm Average Precision cho các mô hình."""
    setup_plot_style()
    df = load_predictions_data(data)

    if model_keys is None:
        available_keys = [
            col.replace("y_prob_", "")
            for col in df.columns
            if col.startswith("y_prob_")
        ]
        preferred_order = ["tree", "nonlinear", "linear", "dummy"]
        model_keys = [k for k in preferred_order if k in available_keys]
        if not model_keys:
            model_keys = available_keys

    y_true_binary = (
        df["y_true_binary"].to_numpy()
        if "y_true_binary" in df.columns
        else (df["y_true"] == pos_label).astype(int).to_numpy()
    )

    pos_ratio = np.mean(y_true_binary == 1)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # Reference horizontal line: No-Skill baseline = positive class prevalence
    ax.axhline(
        y=pos_ratio,
        linestyle="--",
        color="#7f8c8d",
        linewidth=1.5,
        alpha=0.85,
        label=f"No Skill (Prevalence = {pos_ratio:.4f})",
    )

    for key in model_keys:
        prob_col = f"y_prob_{key}"
        pred_col = f"y_pred_{key}"
        if prob_col not in df.columns:
            continue

        y_prob = df[prob_col].to_numpy()
        cfg = DEFAULT_MODEL_CONFIGS.get(key, {})
        name = cfg.get("name", key.capitalize())
        color = cfg.get("color", "#2c3e50")
        linestyle = cfg.get("linestyle", "-")

        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
        ap = average_precision_score(y_true_binary, y_prob)

        lw = 2.5 if key == "tree" else 1.8
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {ap:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            alpha=0.9,
        )

        # Highlight default operating point at tau = 0.5
        if pred_col in df.columns and key != "dummy":
            y_pred = df[pred_col].astype(str)
            tp = np.sum((y_pred == pos_label) & (y_true_binary == 1))
            fp = np.sum((y_pred == pos_label) & (y_true_binary == 0))
            fn = np.sum((y_pred != pos_label) & (y_true_binary == 1))
            op_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            op_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            ax.scatter(
                op_rec,
                op_prec,
                color=color,
                edgecolor="#ffffff",
                s=90,
                zorder=5,
                linewidth=1.5,
            )

    # Explanatory marker
    ax.scatter(
        [], [],
        color="#2c3e50",
        edgecolor="#ffffff",
        s=80,
        linewidth=1.5,
        label=r"Default Operating Point ($\tau = 0.50$)",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity / Detection Rate)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Precision-Recall (PR) Curve Multi-Model Comparison\n"
        f"Target Class: '{pos_label}' (Deposit Subscriber)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        if save_vector:
            fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")

    return fig


# ========================================================================= #
# REGRESSION DIAGNOSTICS (Optional / Auxiliary Tasks)                       #
# ========================================================================= #

def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Regression Model",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Vẽ biểu đồ phân bố phần dư hỗ trợ đánh giá bài toán hồi quy."""
    setup_plot_style()
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    residuals = yt - yp

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(yp, residuals, alpha=0.5, color="#2980b9", edgecolor="none", s=30)
    ax1.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Zero Error")
    ax1.set_xlabel("Predicted Value", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{model_name}: Residuals vs. Predicted", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Distribution of Residuals
    ax2 = axes[1]
    sns.histplot(
        residuals,
        kde=True,
        color="#27ae60",
        ax=ax2,
        stat="density",
        bins=30,
        edgecolor="#ffffff",
        alpha=0.6,
    )
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax2.axvline(mean_res, color="#c0392b", linestyle="--", label=f"Mean = {mean_res:.3f}")
    ax2.set_xlabel("Residual Value", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Residual Distribution (Std = {std_res:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Regression Model",
    output_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Vẽ biểu đồ tương quan giữa giá trị thực tế và giá trị dự đoán hồi quy."""
    setup_plot_style()
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    r2 = r2_score(yt, yp)
    rmse = root_mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(yt, yp, alpha=0.55, color="#8e44ad", edgecolor="none", s=35, label="Test Samples")

    # Identity 45-degree line
    min_val = min(np.min(yt), np.min(yp))
    max_val = max(np.max(yt), np.max(yp))
    margin = 0.05 * (max_val - min_val) if max_val != min_val else 1.0
    line_min = min_val - margin
    line_max = max_val + margin

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#e74c3c",
        linestyle="--",
        linewidth=2.0,
        label=r"Identity Line ($y = \hat{y}$)",
    )

    metric_text = (
        f"$R^2 = {r2:.4f}$\n"
        rf"$\mathrm{{RMSE}} = {rmse:.4f}$" "\n"
        rf"$\mathrm{{MAE}} = {mae:.4f}$"
    )
    ax.text(
        0.05,
        0.92,
        metric_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_xlim([line_min, line_max])
    ax.set_ylim([line_min, line_max])
    ax.set_xlabel("Actual Ground Truth ($y$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Predicted Value ($\hat{y}$)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Actual vs. Predicted Fit Diagnostic: {model_name}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.9)
    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig


# ========================================================================= #
# HIGH-LEVEL PIPELINE EXECUTION                                             #
# ========================================================================= #

def generate_all_diagnostic_plots(
    predictions_path: str | Path | None = None,
    output_dir: str | Path = FIGURES_DIR,
    dpi: int = 300,
    save_vector: bool = True,
) -> dict[str, Path]:
    """Thực thi vẽ và xuất toàn bộ hệ thống đồ thị chẩn đoán đánh giá mô hình."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions_data(predictions_path)

    cm_path = out_dir / "06_confusion_matrices.png"
    roc_path = out_dir / "07_roc_curves.png"
    pr_path = out_dir / "08_pr_curves.png"

    print("Generating Figure 06: Side-by-Side Confusion Matrices...")
    fig_cm = plot_confusion_matrices(df, output_path=cm_path, dpi=dpi, save_vector=save_vector)
    plt.close(fig_cm)

    print("Generating Figure 07: Receiver Operating Characteristic (ROC) Curves...")
    fig_roc = plot_roc_curves(df, output_path=roc_path, dpi=dpi, save_vector=save_vector)
    plt.close(fig_roc)

    print("Generating Figure 08: Precision-Recall (PR) Curves...")
    fig_pr = plot_pr_curves(df, output_path=pr_path, dpi=dpi, save_vector=save_vector)
    plt.close(fig_pr)

    print(f"All diagnostic plots successfully saved to: {out_dir.resolve()}")
    return {
        "confusion_matrices": cm_path,
        "roc_curves": roc_path,
        "pr_curves": pr_path,
    }


if __name__ == "__main__":
    generate_all_diagnostic_plots()
