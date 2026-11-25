"""
Module: preprocess.py
Mô tả: Pipeline tiền xử lý dữ liệu sinh viên.
       - StandardScaler cho numeric features
       - Train-test split stratified
       - SMOTE để xử lý class imbalance (chỉ trên train set)
       - Lưu / tải pipeline bằng joblib
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.data.load_data import FEATURE_COLS, TARGET_COL, LABEL_MAP


# Đường dẫn mặc định
DEFAULT_PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
DEFAULT_DATA_DIR = "data/processed"


def build_pipeline() -> Pipeline:
    """
    Xây dựng sklearn Pipeline chỉ gồm StandardScaler.
    Dataset UCI đã mã hóa số sẵn, không cần One-Hot Encoding.

    Returns
    -------
    Pipeline
        Pipeline chưa fit, sẵn sàng để fit trên X_train.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    return pipeline


def split_raw(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified split trên dữ liệu thô (chưa scale, chưa SMOTE)."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )


def split_and_resample(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    return_raw: bool = False,
):
    """
    Tách train/test với stratified split, sau đó áp dụng SMOTE trên train set.

    ⚠️ SMOTE chỉ được áp dụng trên X_train, KHÔNG trên X_test.

    Parameters
    ----------
    X : pd.DataFrame  — features
    y : pd.Series     — target (đã mã hóa 0/1/2)
    test_size : float — tỷ lệ test (mặc định 0.2)
    random_state : int
    use_smote : bool  — có dùng SMOTE không

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Dữ liệu sau split và (nếu dùng SMOTE) đã resample trên train.
    pipeline : Pipeline
        Pipeline đã fit trên X_train (để dùng lại khi predict).
    """
    # 1. Stratified split
    X_train_raw, X_test_raw, y_train, y_test = split_raw(
        X, y, test_size=test_size, random_state=random_state,
    )

    print(f"Train size: {X_train_raw.shape[0]} | Test size: {X_test_raw.shape[0]}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test  class distribution: {np.bincount(y_test)}")

    # 2. Fit pipeline trên X_train, transform cả train và test
    pipeline = build_pipeline()
    X_train_scaled = pipeline.fit_transform(X_train_raw)
    X_test_scaled  = pipeline.transform(X_test_raw)

    # 3. SMOTE chỉ trên train set
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        print(f"Sau SMOTE — Train: {X_train_res.shape[0]} mẫu | "
              f"Class dist: {np.bincount(y_train_res)}")
    else:
        X_train_res, y_train_res = X_train_scaled, y_train.values

    if return_raw:
        return (X_train_res, X_test_scaled, y_train_res, y_test.values, pipeline,
                X_train_raw, y_train.values)
    return X_train_res, X_test_scaled, y_train_res, y_test.values, pipeline


def save_processed_data(
    X_train, X_test, y_train, y_test,
    feature_names: list,
    output_dir: str = DEFAULT_DATA_DIR,
) -> None:
    """
    Lưu dữ liệu đã xử lý ra file CSV.

    Parameters
    ----------
    X_train, X_test : ndarray — features đã scale
    y_train, y_test : ndarray — nhãn
    feature_names : list      — tên các cột feature
    output_dir : str          — thư mục đầu ra
    """
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_test.csv"), index=False
    )
    pd.Series(y_train, name=TARGET_COL).to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False
    )
    pd.Series(y_test, name=TARGET_COL).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False
    )
    print(f"✅ Đã lưu processed data vào {output_dir}/")


def save_pipeline(pipeline: Pipeline, path: str = DEFAULT_PIPELINE_PATH) -> None:
    """
    Lưu pipeline đã fit bằng joblib.

    Parameters
    ----------
    pipeline : Pipeline — sklearn Pipeline đã fit
    path : str          — đường dẫn file .pkl
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Pipeline đã lưu: {path}")


def load_pipeline(path: str = DEFAULT_PIPELINE_PATH) -> Pipeline:
    """
    Tải pipeline từ file .pkl.

    Parameters
    ----------
    path : str — đường dẫn file .pkl

    Returns
    -------
    Pipeline — sklearn Pipeline đã fit
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy pipeline: {path}")
    pipeline = joblib.load(path)
    print(f"✅ Pipeline đã tải: {path}")
    return pipeline


def run_preprocessing(
    data_path: str = "data/raw/dataset.csv",
    save_data: bool = True,
    use_smote: bool = True,
):
    """
    Hàm tổng hợp: load data → encode → split → SMOTE → save.
    Sử dụng cho notebook 03_Preprocessing.ipynb.

    Returns
    -------
    X_train, X_test, y_train, y_test, pipeline
    """
    from src.data.load_data import load_raw_data, encode_target, get_X_y

    df = load_raw_data(data_path)
    df = encode_target(df)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test, pipeline = split_and_resample(
        X, y, use_smote=use_smote
    )

    if save_data:
        save_processed_data(X_train, X_test, y_train, y_test, FEATURE_COLS)
        save_pipeline(pipeline)

    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    run_preprocessing()
