"""
Module: plots.py
Mô tả: Các hàm vẽ biểu đồ tái sử dụng cho EDA và kết quả mô hình.
       Sử dụng trong notebooks và web app Streamlit.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Style chung
sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = {"Dropout": "#e74c3c", "Enrolled": "#f39c12", "Graduate": "#2ecc71"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHÂN BỐ NHÃN
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "Target",
    save_path: str = None,
    figsize=(7, 5),
) -> plt.Figure:
    """
    Vẽ biểu đồ phân bố nhãn Dropout / Enrolled / Graduate.
    """
    counts = df[target_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    colors = [PALETTE.get(l, "#95a5a6") for l in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for i, (label, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 20, str(cnt), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Số lượng mẫu theo nhãn", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Số sinh viên")
    axes[0].set_ylim(0, counts.max() * 1.15)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Tỷ lệ phân bố nhãn", fontsize=13, fontweight="bold")

    plt.suptitle("Phân bố nhãn Target", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. HISTOGRAM / PHÂN PHỐI ĐẶC TRƯNG
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    features: list,
    target_col: str = "Target",
    ncols: int = 3,
    save_path: str = None,
) -> plt.Figure:
    """
    Vẽ histogram của nhiều đặc trưng, tô màu theo nhãn.
    """
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        for label, color in PALETTE.items():
            subset = df[df[target_col] == label][feat]
            axes[i].hist(subset, bins=20, alpha=0.6, color=color, label=label, density=True)
        axes[i].set_title(feat, fontsize=10)
        axes[i].legend(fontsize=8)
        axes[i].set_ylabel("Density")

    # Ẩn các axes thừa
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Phân phối đặc trưng theo nhãn", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. COUNTPLOT THEO NHÃN
# ─────────────────────────────────────────────────────────────────────────────

def plot_categorical_vs_target(
    df: pd.DataFrame,
    feature: str,
    target_col: str = "Target",
    feature_labels: dict = None,
    save_path: str = None,
    figsize=(8, 5),
) -> plt.Figure:
    """
    Vẽ countplot của 1 biến phân loại, phân tách theo nhãn.

    Parameters
    ----------
    feature_labels : dict — ánh xạ giá trị số → tên (ví dụ {0: 'Nữ', 1: 'Nam'})
    """
    plot_df = df.copy()
    if feature_labels:
        plot_df[feature] = plot_df[feature].map(feature_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        data=plot_df, x=feature, hue=target_col,
        palette=PALETTE, ax=ax, edgecolor="white",
    )
    ax.set_title(f"Phân bố '{feature}' theo nhãn Target", fontsize=13, fontweight="bold")
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel("Số sinh viên", fontsize=11)
    ax.legend(title="Target", fontsize=10)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. HEATMAP TƯƠNG QUAN
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list = None,
    save_path: str = None,
    figsize=(14, 12),
) -> plt.Figure:
    """
    Vẽ correlation heatmap cho các đặc trưng số.
    """
    if features:
        corr_df = df[features].corr()
    else:
        corr_df = df.select_dtypes(include=[np.number]).corr()

    mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Chỉ hiện nửa dưới

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Heatmap tương quan giữa các đặc trưng", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP FEATURES TƯƠNG QUAN VỚI DROPOUT
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_corr_with_dropout(
    df: pd.DataFrame,
    target_col: str = "Target",
    top_n: int = 15,
    save_path: str = None,
    figsize=(9, 6),
) -> plt.Figure:
    """
    Vẽ bar chart top N đặc trưng có tương quan cao nhất với nhãn Dropout.
    """
    # Tạo cột binary: Dropout=1, còn lại=0
    df_tmp = df.copy()
    df_tmp["_is_dropout"] = (df_tmp[target_col] == "Dropout").astype(int)
    numeric_cols = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "_is_dropout"]

    corr = df_tmp[numeric_cols + ["_is_dropout"]].corr()["_is_dropout"].drop("_is_dropout")
    corr_sorted = corr.abs().sort_values(ascending=False).head(top_n)
    corr_vals   = corr[corr_sorted.index]

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(corr_vals.index[::-1], corr_vals.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson Correlation với Dropout", fontsize=11)
    ax.set_title(f"Top {top_n} đặc trưng tương quan với Dropout", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    model_name: str = "Model",
    save_path: str = None,
    figsize=(9, 6),
) -> plt.Figure:
    """
    Vẽ feature importance từ tree-based model (RF, XGBoost, GB).
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Mô hình không có thuộc tính feature_importances_.")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores   = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    ax.barh(top_features[::-1], top_scores[::-1], color=colors)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. BOXPLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    features: list,
    target_col: str = "Target",
    ncols: int = 3,
    save_path: str = None,
) -> plt.Figure:
    """
    Vẽ boxplot của nhiều đặc trưng, phân tách theo nhãn.
    """
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(
            data=df, x=target_col, y=feat,
            palette=PALETTE, ax=axes[i],
        )
        axes[i].set_title(feat, fontsize=10)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplot đặc trưng theo nhãn", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Lưu figure ra file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Đã lưu: {path}")
