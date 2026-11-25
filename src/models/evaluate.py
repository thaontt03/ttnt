"""
Module: evaluate.py
Mô tả: Đánh giá mô hình ML — Classification Report, Confusion Matrix, ROC-AUC.
       Tất cả hàm vẽ biểu đồ đều hỗ trợ lưu file.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, an toàn khi chạy script

import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
)


CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


def print_classification_report(y_true, y_pred) -> None:
    """In classification report đầy đủ ra console."""
    print("\nCLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def get_metrics_dict(y_true, y_pred, y_prob=None) -> dict:
    """
    Tính toán tất cả metric và trả về dict.

    Parameters
    ----------
    y_true : ndarray — nhãn thực
    y_pred : ndarray — nhãn dự đoán
    y_prob : ndarray, shape (n, 3) — xác suất từng lớp (dùng cho AUC)

    Returns
    -------
    dict — accuracy, f1_macro, recall_dropout, precision_macro, auc_macro
    """
    metrics = {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro"), 4),
        # Recall riêng cho lớp Dropout (class 0)
        "recall_dropout":  round(recall_score(y_true, y_pred, labels=[0], average="macro"), 4),
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["auc_macro"] = round(auc, 4)
        except Exception:
            metrics["auc_macro"] = None

    return metrics


def plot_confusion_matrix(
    y_true, y_pred,
    title: str = "Confusion Matrix",
    save_path: str = None,
    figsize=(8, 6),
) -> plt.Figure:
    """
    Vẽ Confusion Matrix dạng heatmap.

    Parameters
    ----------
    y_true, y_pred : ndarray
    title : str    — tiêu đề biểu đồ
    save_path : str — đường dẫn lưu ảnh (None = không lưu)

    Returns
    -------
    plt.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Đã lưu: {save_path}")

    return fig


def plot_roc_curves(
    y_true, y_prob,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: str = None,
    figsize=(8, 6),
) -> plt.Figure:
    """
    Vẽ ROC curve cho từng lớp (One-vs-Rest).

    Parameters
    ----------
    y_true  : ndarray — nhãn thực (0/1/2)
    y_prob  : ndarray, shape (n, 3) — xác suất từng lớp
    """
    from sklearn.preprocessing import label_binarize
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=figsize)

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Đã lưu: {save_path}")

    return fig


def plot_model_comparison(
    df_results: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: str = None,
    figsize=(10, 5),
) -> plt.Figure:
    """
    Vẽ biểu đồ bar so sánh các mô hình theo metric chỉ định.

    Parameters
    ----------
    df_results : pd.DataFrame — bảng so sánh từ train.compare_models()
    metric : str — tên cột metric (mặc định 'f1_macro')
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_results)))
    bars = ax.barh(df_results["Model"], df_results[metric], color=colors)

    for bar, val in zip(bars, df_results[metric]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f"So sánh mô hình — {metric}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Đã lưu: {save_path}")

    return fig


def full_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Best Model",
    figures_dir: str = "reports/figures",
    threshold: float | None = None,
) -> dict:
    """
    Đánh giá toàn diện mô hình trên test set.
    In report, vẽ CM và ROC, lưu ảnh.

    threshold : nếu có, dự đoán = Dropout khi P(Dropout) ≥ threshold (giống web app).

    Returns
    -------
    dict — tất cả metric
    """
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    if threshold is not None and y_prob is not None:
        # Dropout ⇔ P(Dropout) ≥ ngưỡng; ngược lại chọn giữa Enrolled / Graduate
        y_pred = np.where(y_prob[:, 0] >= threshold, 0, 1 + np.argmax(y_prob[:, 1:], axis=1))
    else:
        y_pred = model.predict(X_test)

    print_classification_report(y_test, y_pred)
    metrics = get_metrics_dict(y_test, y_pred, y_prob)

    print(f"\n📊 TỔNG HỢP METRIC ({model_name}):")
    for k, v in metrics.items():
        print(f"   {k:20s}: {v}")

    # Vẽ và lưu biểu đồ
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix — {model_name}",
        save_path=os.path.join(figures_dir, "confusion_matrix.png"),
    )

    if y_prob is not None:
        plot_roc_curves(
            y_test, y_prob,
            title=f"ROC Curves — {model_name}",
            save_path=os.path.join(figures_dir, "roc_curves.png"),
        )

    return metrics
