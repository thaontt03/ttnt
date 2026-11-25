"""
Module: predict.py
Mô tả: Load mô hình đã train và dự đoán sinh viên mới (đơn lẻ, hàng loạt, nhiều mô hình),
       kèm giải thích cục bộ theo nhóm yếu tố. Được web app Streamlit dùng trực tiếp.

Quy ước:
  - Kết luận chính: "Có nguy cơ bỏ học" ⇔ P(Dropout) ≥ ngưỡng của CHÍNH mô hình đó
    (ngưỡng chọn trên dự đoán out-of-fold lúc train, lưu ở predict_config.json).
    Lớp dự đoán 3 lớp nhất quán: Dropout nếu vượt ngưỡng, ngược lại Enrolled/Graduate.
  - Mức rủi ro được tính tương đối theo ngưỡng nên luôn nhất quán với kết luận:
        Thấp         : p < 0.7·t
        Cần theo dõi : 0.7·t ≤ p < t
        Cao          : t ≤ p < (1 + t)/2
        Rất cao      : p ≥ (1 + t)/2
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import FEATURE_COLS, LABEL_MAP_INV

CLASS_NAMES = [LABEL_MAP_INV[i] for i in range(3)]          # Dropout, Enrolled, Graduate
DROPOUT_THRESHOLD_DEFAULT = 0.35

RISK_LEVELS = ["Thấp", "Cần theo dõi", "Cao", "Rất cao"]     # tăng dần
AT_RISK_LEVELS = {"Cao", "Rất cao"}
LABEL_AT_RISK = "Có nguy cơ bỏ học"
LABEL_SAFE = "Không có nguy cơ"

MODEL_ORDER = [
    "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees",
    "SVM", "KNN", "Gradient Boosting", "AdaBoost", "XGBoost", "LightGBM",
]

DEFAULT_MODEL_PATH = "models/best_model.pkl"
DEFAULT_PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
DEFAULT_CONFIG_PATH = "models/predict_config.json"

# Nhóm đặc trưng cho phần giải thích (thay cả nhóm bằng giá trị điển hình)
FEATURE_GROUPS = {
    "Kết quả học kỳ 1": [c for c in FEATURE_COLS if "1st sem" in c],
    "Kết quả học kỳ 2": [c for c in FEATURE_COLS if "2nd sem" in c],
    "Đóng học phí đúng hạn": ["Tuition fees up to date"],
    "Nợ học phí": ["Debtor"],
    "Học bổng": ["Scholarship holder"],
    "Tuổi nhập học": ["Age at enrollment"],
    "Ngành & ca học": ["Course", "Daytime/evening attendance"],
    "Hình thức tuyển sinh": ["Application mode", "Application order"],
    "Điểm đầu vào": ["Admission grade", "Previous qualification (grade)"],
    "Trình độ trước khi nhập học": ["Previous qualification"],
    "Hoàn cảnh gia đình": ["Mother's qualification", "Father's qualification",
                           "Mother's occupation", "Father's occupation"],
    "Nhân khẩu học": ["Marital status", "Gender", "Nacionality", "International",
                      "Displaced", "Educational special needs"],
    "Kinh tế vĩ mô": ["Unemployment rate", "Inflation rate", "GDP"],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def model_filename(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".pkl"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    config = {"dropout_threshold": DROPOUT_THRESHOLD_DEFAULT, "thresholds": {}}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config.update(json.load(f))
    return config


def get_threshold(config: dict | None, model_name: str | None = None) -> float:
    """Ngưỡng riêng của mô hình nếu có, ngược lại ngưỡng mặc định của config."""
    config = config or {}
    per_model = config.get("thresholds") or {}
    if model_name and model_name in per_model:
        return float(per_model[model_name])
    return float(config.get("dropout_threshold", DROPOUT_THRESHOLD_DEFAULT))


def list_available_models(models_dir: str = "models") -> list[str]:
    """Chỉ liệt kê các mô hình THỰC SỰ có file .pkl riêng."""
    return [m for m in MODEL_ORDER
            if os.path.exists(os.path.join(models_dir, model_filename(m)))]


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
):
    """Load (model, pipeline, config)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình: {model_path}\n"
                                "Hãy chạy `python run_pipeline.py` để train trước.")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Không tìm thấy pipeline: {pipeline_path}\n"
                                "Hãy chạy `python run_pipeline.py` trước.")
    return joblib.load(model_path), joblib.load(pipeline_path), load_config(config_path)


def load_named_model(
    model_name: str,
    models_dir: str = "models",
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
):
    """Load mô hình theo tên; báo lỗi rõ ràng nếu thiếu file (không âm thầm thay bằng best_model)."""
    path = os.path.join(models_dir, model_filename(model_name))
    return load_model(path, pipeline_path, config_path)


# ─────────────────────────────────────────────────────────────────────────────
# CHUẨN HOÁ ĐẦU VÀO
# ─────────────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return " ".join(str(name).replace("﻿", "").split()).lower()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Chuẩn hoá DataFrame đầu vào về đúng 36 cột FEATURE_COLS (đúng thứ tự, kiểu số).

    - Khớp tên cột không phân biệt hoa/thường, khoảng trắng/tab thừa, BOM.
    - Ép kiểu số; trả về danh sách chỉ số dòng có giá trị thiếu/không hợp lệ.

    Raises
    ------
    ValueError nếu thiếu cột bắt buộc.
    """
    lookup = {_norm(c): c for c in df.columns}
    missing = [c for c in FEATURE_COLS if _norm(c) not in lookup]
    if missing:
        raise ValueError(f"Thiếu {len(missing)} cột bắt buộc: " + ", ".join(missing))
    X = pd.DataFrame({c: pd.to_numeric(df[lookup[_norm(c)]], errors="coerce")
                      for c in FEATURE_COLS}, index=df.index)
    bad_rows = X.index[X.isna().any(axis=1)].tolist()
    return X, bad_rows


def _to_frame(student_data: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(student_data, pd.DataFrame):
        return student_data[FEATURE_COLS].astype(float)
    return pd.DataFrame([dict(student_data)])[FEATURE_COLS].astype(float)


def predict_proba(model, pipeline, X: pd.DataFrame) -> np.ndarray:
    """Xác suất 3 lớp (Dropout, Enrolled, Graduate)."""
    X_scaled = pipeline.transform(X[FEATURE_COLS])
    return model.predict_proba(X_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# KẾT LUẬN & MỨC RỦI RO
# ─────────────────────────────────────────────────────────────────────────────

def risk_level(p_dropout: float, threshold: float) -> str:
    if p_dropout >= (1 + threshold) / 2:
        return "Rất cao"
    if p_dropout >= threshold:
        return "Cao"
    if p_dropout >= 0.7 * threshold:
        return "Cần theo dõi"
    return "Thấp"


def risk_levels(p: np.ndarray, threshold: float) -> np.ndarray:
    p = np.asarray(p)
    out = np.full(p.shape, "Thấp", dtype=object)
    out[p >= 0.7 * threshold] = "Cần theo dõi"
    out[p >= threshold] = "Cao"
    out[p >= (1 + threshold) / 2] = "Rất cao"
    return out


def _result_from_proba(proba: np.ndarray, threshold: float, model_name: str | None) -> dict:
    p = float(proba[0])
    at_risk = p >= threshold
    pred_cls = 0 if at_risk else 1 + int(np.argmax(proba[1:]))
    return {
        "model_name":      model_name,
        "at_risk":         bool(at_risk),
        "at_risk_label":   LABEL_AT_RISK if at_risk else LABEL_SAFE,
        "dropout_prob":    round(p, 4),
        "threshold":       round(float(threshold), 4),
        "margin":          round(p - threshold, 4),
        "probabilities":   {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(3)},
        "predicted_class": CLASS_NAMES[pred_cls],
        "risk_level":      risk_level(p, threshold),
        "error":           None,
    }


def predict_single(
    student_data: dict,
    model=None,
    pipeline=None,
    config: dict | None = None,
    model_name: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
) -> dict:
    """
    Dự đoán cho 1 sinh viên.

    Returns dict: model_name, at_risk, at_risk_label, dropout_prob, threshold, margin,
                  probabilities{Dropout,Enrolled,Graduate}, predicted_class, risk_level, error
    """
    if model is None or pipeline is None:
        model, pipeline, config = load_model(model_path, pipeline_path)
    if model_name is None and config:
        model_name = config.get("best_model")
    proba = predict_proba(model, pipeline, _to_frame(student_data))[0]
    return _result_from_proba(proba, get_threshold(config, model_name), model_name)


def predict_multi_model(
    student_data: dict,
    model_names: list[str],
    models: dict | None = None,
    pipeline=None,
    config: dict | None = None,
    models_dir: str = "models",
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> list[dict]:
    """
    Dự đoán với nhiều mô hình. Truyền `models` (dict tên → model đã load) và `pipeline`
    để không phải đọc lại file .pkl mỗi lần.
    """
    if pipeline is None:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Không tìm thấy pipeline: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
    if config is None:
        config = load_config(config_path)
    X_scaled = pipeline.transform(_to_frame(student_data))

    results = []
    for name in model_names:
        try:
            model = (models or {}).get(name)
            if model is None:
                path = os.path.join(models_dir, model_filename(name))
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Chưa có file {model_filename(name)}")
                model = joblib.load(path)
            proba = model.predict_proba(X_scaled)[0]
            results.append(_result_from_proba(proba, get_threshold(config, name), name))
        except Exception as e:  # noqa: BLE001 — báo lỗi từng mô hình, không dừng cả loạt
            results.append({"model_name": name, "error": str(e)})
    return results


def consensus(results: list[dict]) -> dict:
    """
    Tổng hợp nhiều mô hình: đa số phiếu; nếu hoà thì dùng biên trung bình (P − ngưỡng).
    """
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {}
    votes = sum(r["at_risk"] for r in valid)
    n = len(valid)
    mean_margin = float(np.mean([r["margin"] for r in valid]))
    at_risk = votes * 2 > n or (votes * 2 == n and mean_margin >= 0)
    agree = votes if at_risk else n - votes
    return {
        "at_risk": bool(at_risk),
        "votes_at_risk": int(votes),
        "n_models": n,
        "agreement": agree / n,
        "mean_prob": float(np.mean([r["dropout_prob"] for r in valid])),
        "mean_threshold": float(np.mean([r["threshold"] for r in valid])),
        "mean_margin": mean_margin,
        "tie_broken": votes * 2 == n,
    }


def predict_batch(
    df: pd.DataFrame,
    model=None,
    pipeline=None,
    config: dict | None = None,
    model_name: str | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
) -> pd.DataFrame:
    """
    Dự đoán hàng loạt. `df` phải có đủ FEATURE_COLS (dùng prepare_features() trước nếu cần).

    Returns: df gốc + At_Risk, At_Risk_Label, P_Dropout, P_Enrolled, P_Graduate,
             Predicted_Class, Risk_Level, Threshold
    """
    if model is None or pipeline is None:
        model, pipeline, config = load_model(model_path, pipeline_path)
    if model_name is None and config:
        model_name = config.get("best_model")
    threshold = get_threshold(config, model_name)

    proba = predict_proba(model, pipeline, df)
    p_drop = proba[:, 0]
    at_risk = p_drop >= threshold
    pred = np.where(at_risk, 0, 1 + np.argmax(proba[:, 1:], axis=1))

    result = df.copy()
    result["At_Risk"] = at_risk.astype(int)
    result["At_Risk_Label"] = np.where(at_risk, LABEL_AT_RISK, LABEL_SAFE)
    result["P_Dropout"] = p_drop.round(4)
    result["P_Enrolled"] = proba[:, 1].round(4)
    result["P_Graduate"] = proba[:, 2].round(4)
    result["Predicted_Class"] = [CLASS_NAMES[i] for i in pred]
    result["Risk_Level"] = risk_levels(p_drop, threshold)
    result["Threshold"] = threshold
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GIẢI THÍCH CỤC BỘ
# ─────────────────────────────────────────────────────────────────────────────

def explain_by_groups(
    student_data: dict,
    model,
    pipeline,
    reference: pd.Series | dict,
    groups: dict[str, list[str]] = FEATURE_GROUPS,
) -> pd.DataFrame:
    """
    Giải thích bằng giá trị Shapley CHÍNH XÁC theo nhóm đặc trưng (không phụ thuộc loại mô hình).

    Trò chơi: v(S) = P(Dropout) khi các nhóm trong S lấy giá trị của sinh viên, các nhóm còn lại
    lấy giá trị điển hình (`reference`). Với G nhóm, duyệt đủ 2^G tổ hợp (13 nhóm → 8.192 dòng,
    dự đoán trong 1 lô). Shapley chia công bằng phần chênh lệch kể cả khi các nhóm trùng thông tin
    (ví dụ HK1 và HK2 cùng kém), và luôn thoả:  Σ impact = P(sinh viên) − P(hồ sơ điển hình).

      impact > 0 → nhóm này LÀM TĂNG nguy cơ so với mức điển hình; < 0 → giúp GIẢM.

    Returns DataFrame [group, impact, p_if_typical] sắp theo |impact| giảm dần;
    attrs: p (xác suất của sinh viên), p_ref (xác suất của hồ sơ điển hình).
    `p_if_typical` = P nếu riêng nhóm đó trở về mức điển hình (để tham khảo).
    """
    from math import factorial

    base = _to_frame(student_data).iloc[0]
    reference = pd.Series(reference).reindex(FEATURE_COLS).astype(float)
    names = list(groups)
    G = len(names)
    col_idx = [[FEATURE_COLS.index(c) for c in groups[g]] for g in names]

    masks = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1).astype(bool)   # (2^G, G)
    X = np.tile(reference.values, (2 ** G, 1))
    for j, idx in enumerate(col_idx):
        rows = masks[:, j]
        X[np.ix_(rows, idx)] = base.values[idx]
    v = predict_proba(model, pipeline, pd.DataFrame(X, columns=FEATURE_COLS))[:, 0]

    size = masks.sum(axis=1)
    w = np.array([factorial(s) * factorial(G - s - 1) / factorial(G) for s in range(G)])
    full = 2 ** G - 1
    phi, p_alone = np.zeros(G), np.zeros(G)
    for j in range(G):
        without = np.where(~masks[:, j])[0]
        with_j = without | (1 << j)
        phi[j] = np.sum(w[size[without]] * (v[with_j] - v[without]))
        p_alone[j] = v[full & ~(1 << j)]

    out = pd.DataFrame({"group": names, "impact": phi, "p_if_typical": p_alone})
    out = (out.reindex(out["impact"].abs().sort_values(ascending=False).index)
              .reset_index(drop=True))
    out.attrs["p"] = float(v[full])
    out.attrs["p_ref"] = float(v[0])
    return out


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample = pd.read_csv(os.path.join(ROOT, "data", "raw", "dataset.csv"), sep=";").iloc[0]
    X, _ = prepare_features(sample.to_frame().T)
    m, pl, cfg = load_model(os.path.join(ROOT, DEFAULT_MODEL_PATH),
                            os.path.join(ROOT, DEFAULT_PIPELINE_PATH),
                            os.path.join(ROOT, DEFAULT_CONFIG_PATH))
    print(predict_single(X.iloc[0].to_dict(), m, pl, cfg))
