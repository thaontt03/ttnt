"""
Module: train.py
Mô tả: Huấn luyện, đánh giá trung thực và so sánh 10 thuật toán ML.

Quy trình chuẩn (tránh rò rỉ dữ liệu):
  1. CV StratifiedKFold trên tập train GỐC, SMOTE + scaler nằm TRONG từng fold
     (imblearn Pipeline) → số liệu CV phản ánh đúng phân bố thực tế.
  2. Dự đoán out-of-fold (OOF) dùng để chọn ngưỡng Dropout riêng cho từng mô hình
     → không dùng tập test để tuning.
  3. Fit mô hình cuối trên toàn bộ train (scale → SMOTE) rồi đánh giá 1 lần trên test.
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGBM_AVAILABLE = False


RANDOM_STATE = 42
DROPOUT = 0                      # mã lớp Dropout
RECALL_TARGET = 0.75             # mục tiêu Recall(Dropout) của dự án
TUNING_MARGIN = 0.03             # biên an toàn khi chọn ngưỡng trên OOF (dao động giữa các fold ≈ 0.02–0.03)
THRESHOLD_GRID = np.round(np.arange(0.10, 0.81, 0.01), 2)


# ── Danh sách mô hình ──────────────────────────────────────────────────────────

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", min_samples_leaf=5, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
        random_state=RANDOM_STATE,
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    ),
}

if _LGBM_AVAILABLE:
    MODELS["LightGBM"] = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

# ── Grid Search param grids (tuỳ chọn) ─────────────────────────────────────────

PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"n_estimators": [200, 400], "max_depth": [None, 10, 20],
                      "min_samples_leaf": [1, 2, 4]},
    "Extra Trees": {"n_estimators": [200, 400], "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 2, 4]},
    "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [0.3, 0.5, 1.0]},
    "XGBoost": {"n_estimators": [200, 300], "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1]},
    "LightGBM": {"n_estimators": [200, 300], "num_leaves": [15, 31],
                 "learning_rate": [0.03, 0.05, 0.1]},
    "Gradient Boosting": {"n_estimators": [100, 200], "max_depth": [3, 5],
                          "learning_rate": [0.05, 0.1]},
    "SVM": {"C": [0.3, 1, 3], "gamma": ["scale", "auto"]},
}


def model_filename(model_name: str) -> str:
    """'Random Forest' → 'random_forest.pkl'."""
    return model_name.lower().replace(" ", "_") + ".pkl"


# ─────────────────────────────────────────────────────────────────────────────
# NGƯỠNG DROPOUT & METRIC
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Dự đoán 3 lớp nhất quán với kết luận nhị phân:
      Dropout ⇔ P(Dropout) ≥ ngưỡng; ngược lại chọn lớp lớn hơn giữa Enrolled / Graduate.
    """
    proba = np.asarray(proba)
    return np.where(proba[:, DROPOUT] >= threshold, DROPOUT,
                    1 + np.argmax(proba[:, 1:], axis=1))


def compute_metrics(y_true, proba: np.ndarray, threshold: float | None) -> dict:
    """Bộ metric đầy đủ tại một ngưỡng Dropout (threshold=None → argmax thuần)."""
    y_pred = np.argmax(proba, axis=1) if threshold is None else apply_threshold(proba, threshold)
    is_drop = (np.asarray(y_true) == DROPOUT).astype(int)
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        auc_drop = roc_auc_score(is_drop, proba[:, DROPOUT])
    except ValueError:
        auc, auc_drop = np.nan, np.nan
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro"),
        "recall_macro":      recall_score(y_true, y_pred, average="macro"),
        "precision_macro":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_dropout":    recall_score(y_true, y_pred, labels=[DROPOUT], average="macro"),
        "precision_dropout": precision_score(y_true, y_pred, labels=[DROPOUT],
                                             average="macro", zero_division=0),
        "f1_dropout":        f1_score(y_true, y_pred, labels=[DROPOUT], average="macro"),
        "auc_macro":         auc,
        "auc_dropout":       auc_drop,
    }


def tune_threshold(
    y_true, proba: np.ndarray,
    recall_target: float = RECALL_TARGET + TUNING_MARGIN,
    grid: np.ndarray = THRESHOLD_GRID,
) -> float:
    """
    Chọn ngưỡng Dropout trên dự đoán out-of-fold:
      - Ưu tiên các ngưỡng đạt Recall(Dropout) ≥ recall_target (= mục tiêu + biên an toàn), chọn ngưỡng có F1-macro cao nhất
        (hoà thì chọn ngưỡng cho Recall cao hơn).
      - Nếu không ngưỡng nào đạt, chọn ngưỡng tối đa 0.6·Recall + 0.4·F1-macro.
    """
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        y_pred = apply_threshold(proba, t)
        rec = recall_score(y_true, y_pred, labels=[DROPOUT], average="macro")
        f1m = f1_score(y_true, y_pred, average="macro")
        rows.append((t, rec, f1m))
    feasible = [r for r in rows if r[1] >= recall_target]
    if feasible:
        best = max(feasible, key=lambda r: (round(r[2], 4), r[1]))
    else:
        best = max(rows, key=lambda r: 0.6 * r[1] + 0.4 * r[2])
    return float(best[0])


# ─────────────────────────────────────────────────────────────────────────────
# ĐÁNH GIÁ TRUNG THỰC
# ─────────────────────────────────────────────────────────────────────────────

def make_cv_pipeline(model) -> ImbPipeline:
    """Scaler + SMOTE + model — SMOTE chỉ chạy trên phần train của mỗi fold."""
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", clone(model)),
    ])


def oof_predict(model, X_raw: pd.DataFrame, y: np.ndarray, cv: int = 5):
    """Xác suất out-of-fold + chỉ số fold của từng mẫu."""
    X_raw = np.asarray(X_raw)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), 3))
    fold_id = np.zeros(len(y), dtype=int)
    for k, (tr, va) in enumerate(skf.split(X_raw, y)):
        pipe = make_cv_pipeline(model)
        pipe.fit(X_raw[tr], y[tr])
        oof[va] = pipe.predict_proba(X_raw[va])
        fold_id[va] = k
    return oof, fold_id


def evaluate_model(
    name: str,
    model,
    X_train_raw, y_train,
    X_test_scaled, y_test,
    X_train_res, y_train_res,
    cv: int = 5,
):
    """
    Đánh giá 1 mô hình theo quy trình chuẩn.

    Returns
    -------
    row : dict      — metric CV (OOF, tại ngưỡng tối ưu) + metric test
    fitted : model  — mô hình đã fit trên toàn bộ train (scale + SMOTE)
    """
    y_train = np.asarray(y_train)
    oof, fold_id = oof_predict(model, X_train_raw, y_train, cv=cv)
    threshold = tune_threshold(y_train, oof)

    cv_m = compute_metrics(y_train, oof, threshold)
    fold_f1 = [
        f1_score(y_train[fold_id == k], apply_threshold(oof[fold_id == k], threshold),
                 average="macro")
        for k in range(cv)
    ]

    fitted = clone(model)
    fitted.fit(X_train_res, y_train_res)
    test_proba = fitted.predict_proba(X_test_scaled)
    test_m = compute_metrics(y_test, test_proba, threshold)
    test_argmax = compute_metrics(y_test, test_proba, threshold=None)  # argmax thuần

    row = {"Model": name, "threshold": threshold}
    row.update({k: round(float(v), 4) for k, v in cv_m.items()})
    row["std_f1"] = round(float(np.std(fold_f1)), 4)
    row.update({f"test_{k}": round(float(v), 4) for k, v in test_m.items()})
    row["test_f1_macro_argmax"] = round(float(test_argmax["f1_macro"]), 4)
    row["test_recall_dropout_argmax"] = round(float(test_argmax["recall_dropout"]), 4)
    return row, fitted


def evaluate_all_models(
    X_train_raw, y_train,
    X_test_scaled, y_test,
    X_train_res, y_train_res,
    models: dict | None = None,
    cv: int = 5,
    verbose: bool = True,
):
    """Đánh giá toàn bộ mô hình. Trả về (DataFrame so sánh, dict mô hình đã fit)."""
    models = models or MODELS
    rows, fitted = [], {}
    for name, model in models.items():
        if verbose:
            print(f"  • {name:20s}", end=" ", flush=True)
        row, m = evaluate_model(name, model, X_train_raw, y_train,
                                X_test_scaled, y_test, X_train_res, y_train_res, cv=cv)
        rows.append(row)
        fitted[name] = m
        if verbose:
            print(f"ngưỡng={row['threshold']:.2f}  CV F1={row['f1_macro']:.4f}  "
                  f"CV Recall(D)={row['recall_dropout']:.4f}  |  "
                  f"Test F1={row['test_f1_macro']:.4f}  Test Recall(D)={row['test_recall_dropout']:.4f}")
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df, fitted


def select_best_model(df_results: pd.DataFrame, recall_target: float = RECALL_TARGET) -> str:
    """Chọn mô hình tốt nhất CHỈ dựa trên số liệu CV (không nhìn tập test)."""
    ok = df_results[df_results["recall_dropout"] >= recall_target]
    pool = ok if len(ok) else df_results
    return str(pool.sort_values(["f1_macro", "auc_macro"], ascending=False).iloc[0]["Model"])


# ─────────────────────────────────────────────────────────────────────────────
# API CŨ (giữ tương thích với notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_cv(model, X_train, y_train, cv: int = 5) -> dict:
    """
    CV đơn giản trên dữ liệu đã truyền vào.
    ⚠️ Nếu X_train đã SMOTE, kết quả sẽ lạc quan quá mức — dùng evaluate_all_models().
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(model, X_train, y_train, cv=skf, n_jobs=-1,
                         scoring={"accuracy": "accuracy", "f1_macro": "f1_macro",
                                  "recall_macro": "recall_macro"})
    return {
        "accuracy":     round(np.mean(res["test_accuracy"]), 4),
        "f1_macro":     round(np.mean(res["test_f1_macro"]), 4),
        "recall_macro": round(np.mean(res["test_recall_macro"]), 4),
        "std_f1":       round(np.std(res["test_f1_macro"]), 4),
    }


def compare_models(X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """So sánh nhanh (API cũ). Xem cảnh báo ở train_with_cv()."""
    rows = []
    for name, model in MODELS.items():
        print(f"  Training {name}...", end=" ", flush=True)
        metrics = train_with_cv(model, X_train, y_train, cv=cv)
        rows.append({"Model": name, **metrics})
        print(f"F1={metrics['f1_macro']:.4f}")
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def tune_hyperparams(model_name: str, X_train, y_train, cv: int = 5):
    """GridSearchCV (SMOTE nằm trong fold) cho mô hình được chọn."""
    if model_name not in MODELS:
        raise ValueError(f"Không có mô hình '{model_name}' trong MODELS.")
    if model_name not in PARAM_GRIDS:
        model = clone(MODELS[model_name])
        model.fit(X_train, y_train)
        return model
    pipe = ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)),
                        ("model", clone(MODELS[model_name]))])
    grid = {f"model__{k}": v for k, v in PARAM_GRIDS[model_name].items()}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"   Best params : {gs.best_params_}")
    print(f"   Best F1     : {gs.best_score_:.4f}")
    return gs.best_estimator_.named_steps["model"]


def train_best_model(X_train, y_train, model_name: str = "XGBoost", tune: bool = True,
                     save_path: str = "models/best_model.pkl"):
    """Train và lưu mô hình (API cũ)."""
    if tune and model_name in PARAM_GRIDS:
        model = tune_hyperparams(model_name, X_train, y_train)
    else:
        model = clone(MODELS[model_name])
        model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ Mô hình {model_name} đã lưu: {save_path}")
    return model


def save_comparison(df: pd.DataFrame, path: str = "models/model_comparison.csv") -> None:
    """Lưu bảng so sánh ra CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Bảng so sánh lưu: {path}")
