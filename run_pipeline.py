"""
run_pipeline.py
===============
Chạy toàn bộ pipeline huấn luyện theo quy trình chuẩn (không rò rỉ dữ liệu):

  Step 1 → Load & kiểm tra dữ liệu
  Step 2 → Tách train/test (stratified) · fit scaler trên train · SMOTE chỉ trên train
  Step 3 → Mỗi mô hình: CV 5-fold (SMOTE trong fold) → dự đoán out-of-fold
           → chọn ngưỡng Dropout riêng → fit cuối → đánh giá 1 lần trên test
  Step 4 → Chọn mô hình tốt nhất CHỈ theo số liệu CV
  Step 5 → Lưu model, pipeline, bảng so sánh, config ngưỡng, biểu đồ

Dùng:
  python run_pipeline.py                    # tất cả 10 mô hình
  python run_pipeline.py --models XGBoost "Random Forest"
  python run_pipeline.py --cv 3             # nhanh hơn
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402

from src.data.load_data import (  # noqa: E402
    FEATURE_COLS, basic_info, check_data_quality, encode_target, get_X_y, load_raw_data,
)
from src.data.preprocess import save_pipeline, save_processed_data, split_and_resample  # noqa: E402
from src.models.evaluate import full_evaluation  # noqa: E402
from src.models.train import (  # noqa: E402
    MODELS, RECALL_TARGET, evaluate_all_models, model_filename, save_comparison,
    select_best_model,
)

MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "raw", "dataset.csv")


def banner(n, title):
    print(f"\n{'=' * 64}\n  STEP {n}: {title}\n{'=' * 64}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự đoán bỏ học")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Tên mô hình cần train (mặc định: tất cả)")
    parser.add_argument("--cv", type=int, default=5, help="Số fold CV (mặc định 5)")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    models = MODELS
    if args.models:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            parser.error(f"Không có mô hình: {unknown}. Hợp lệ: {list(MODELS)}")
        models = {m: MODELS[m] for m in args.models}

    t0 = time.time()

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    banner(1, "LOAD & KIỂM TRA DỮ LIỆU")
    df = load_raw_data(DATA_PATH)
    basic_info(df)
    check_data_quality(df)
    X, y = get_X_y(encode_target(df))

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    banner(2, "SPLIT · SCALE · SMOTE (chỉ trên train)")
    (X_train_res, X_test, y_train_res, y_test, pipeline,
     X_train_raw, y_train_raw) = split_and_resample(
        X, y, test_size=args.test_size, use_smote=True, return_raw=True)
    save_processed_data(X_train_res, X_test, y_train_res, y_test, FEATURE_COLS,
                        output_dir=os.path.join(ROOT, "data", "processed"))
    save_pipeline(pipeline, path=os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    banner(3, f"ĐÁNH GIÁ {len(models)} MÔ HÌNH (CV {args.cv}-fold, SMOTE trong fold)")
    df_results, fitted = evaluate_all_models(
        X_train_raw, y_train_raw, X_test, y_test, X_train_res, y_train_res,
        models=models, cv=args.cv,
    )

    # Ghép với kết quả cũ nếu chỉ train một phần
    cmp_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    config_path = os.path.join(MODELS_DIR, "predict_config.json")
    old_cfg = {}
    if args.models and os.path.exists(cmp_path):
        import pandas as pd
        old = pd.read_csv(cmp_path)
        if "threshold" in old.columns:
            old = old[~old["Model"].isin(df_results["Model"])]
            df_results = (pd.concat([df_results, old], ignore_index=True)
                          .sort_values("f1_macro", ascending=False).reset_index(drop=True))
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                old_cfg = json.load(f)

    # ── STEP 4 ────────────────────────────────────────────────────────────────
    banner(4, "CHỌN MÔ HÌNH TỐT NHẤT (theo CV)")
    best_name = select_best_model(df_results)
    best_row = df_results.set_index("Model").loc[best_name]
    print(f"🏆 {best_name}  — ngưỡng {best_row['threshold']:.2f}, "
          f"CV F1-macro {best_row['f1_macro']:.4f}, CV Recall(D) {best_row['recall_dropout']:.4f}")

    # ── STEP 5 ────────────────────────────────────────────────────────────────
    banner(5, "LƯU KẾT QUẢ")
    for name, model in fitted.items():
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename(name)))
        print(f"  ✅ {model_filename(name)}")
    best_model = fitted.get(best_name) or joblib.load(
        os.path.join(MODELS_DIR, model_filename(best_name)))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    save_comparison(df_results, path=cmp_path)

    thresholds = dict(old_cfg.get("thresholds", {}))
    thresholds.update({r.Model: float(r.threshold) for r in df_results.itertuples()})
    config = {
        "dropout_threshold": float(best_row["threshold"]),
        "best_model": best_name,
        "thresholds": thresholds,
        "recall_target": RECALL_TARGET,
        "cv_folds": args.cv,
        "test_size": args.test_size,
        "n_train": int(len(y_train_raw)),
        "n_test": int(len(y_test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {os.path.relpath(config_path, ROOT)}")

    figures_dir = os.path.join(ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    metrics = full_evaluation(best_model, X_test, y_test, model_name=best_name,
                              figures_dir=figures_dir, threshold=config["dropout_threshold"])

    print("\n📋 KẾT QUẢ TEST (mô hình tốt nhất):")
    for k, target in [("recall_dropout", RECALL_TARGET), ("f1_macro", 0.70)]:
        v = metrics.get(k)
        print(f"  {'✅' if v is not None and v >= target else '⚠️'} {k:16}: {v}  (mục tiêu ≥ {target})")
    print(f"\n⏱️  Tổng thời gian: {time.time() - t0:.0f}s")
    print("✅ Hoàn tất! Chạy web app: streamlit run app/app.py")


if __name__ == "__main__":
    main()
