"""
Module: load_data.py
Mô tả: Đọc dataset sinh viên từ UCI và kiểm tra chất lượng dữ liệu cơ bản.
Dataset: Predict Students' Dropout and Academic Success
         https://archive.ics.uci.edu/dataset/697
"""

import os
import pandas as pd
import numpy as np


# Tên cột target và mapping nhãn
TARGET_COL = "Target"
LABEL_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
LABEL_MAP_INV = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# Danh sách tất cả 36 đặc trưng (không gồm Target)
FEATURE_COLS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Đọc file CSV dataset và thực hiện kiểm tra cơ bản.

    Parameters
    ----------
    path : str
        Đường dẫn đến file CSV (ví dụ: 'data/raw/dataset.csv')

    Returns
    -------
    pd.DataFrame
        DataFrame với cột Target đã được giữ nguyên dạng string.

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại tại đường dẫn chỉ định.
    ValueError
        Nếu thiếu cột Target hoặc các cột đặc trưng bắt buộc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại: {path}\n"
            "Hãy tải dataset từ https://archive.ics.uci.edu/dataset/697 "
            "và lưu vào data/raw/dataset.csv"
        )

    # Dataset UCI dùng dấu chấm phẩy làm separator
    df = pd.read_csv(path, sep=";")

    # Strip whitespace/tab khỏi tên cột (dataset UCI có trailing tab)
    df.columns = df.columns.str.strip()

    # Kiểm tra cột Target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{TARGET_COL}' trong dataset.")

    print(f"✅ Đọc thành công: {df.shape[0]} mẫu, {df.shape[1]} cột")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """In thông tin cơ bản về dataset."""
    print("=" * 60)
    print("THÔNG TIN DATASET")
    print("=" * 60)
    print(f"Shape        : {df.shape}")
    print(f"Số cột       : {df.shape[1]}")
    print(f"Số mẫu       : {df.shape[0]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()
    print("PHÂN BỐ NHÃN (Target):")
    counts = df[TARGET_COL].value_counts()
    for label, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"  {label:10s}: {cnt:5d}  ({pct:.1f}%)")
    print()
    print("KIỂU DỮ LIỆU:")
    print(df.dtypes.value_counts().to_string())
    print("=" * 60)


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Kiểm tra chất lượng dữ liệu: missing values, duplicates, outliers.

    Returns
    -------
    dict
        Dictionary chứa các thống kê chất lượng dữ liệu.
    """
    quality_report = {}

    # Missing values
    missing = df.isnull().sum()
    quality_report["missing_total"] = int(missing.sum())
    quality_report["missing_cols"] = missing[missing > 0].to_dict()

    # Duplicates
    quality_report["duplicates"] = int(df.duplicated().sum())

    # Unique values per column
    quality_report["unique_values"] = df.nunique().to_dict()

    # Class distribution
    quality_report["class_distribution"] = df[TARGET_COL].value_counts().to_dict()

    print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU:")
    print(f"  Missing values  : {quality_report['missing_total']}")
    print(f"  Duplicate rows  : {quality_report['duplicates']}")
    print(f"  Phân bố nhãn    : {quality_report['class_distribution']}")

    return quality_report


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa cột Target từ string sang số nguyên.
    Dropout=0, Enrolled=1, Graduate=2

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có cột Target dạng string.

    Returns
    -------
    pd.DataFrame
        DataFrame với Target đã mã hóa.
    """
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map(LABEL_MAP)

    if df[TARGET_COL].isnull().any():
        unknown = df[df[TARGET_COL].isnull()]["Target"].unique()
        raise ValueError(f"Nhãn không xác định trong Target: {unknown}")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"✅ Đã mã hóa Target: Dropout=0, Enrolled=1, Graduate=2")
    return df


def get_X_y(df: pd.DataFrame):
    """
    Tách DataFrame thành features X và target y.

    Returns
    -------
    X : pd.DataFrame  — 36 đặc trưng
    y : pd.Series     — nhãn đã mã hóa (0/1/2)
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


if __name__ == "__main__":
    # Test nhanh
    DATA_PATH = "data/raw/dataset.csv"
    df = load_raw_data(DATA_PATH)
    basic_info(df)
    check_data_quality(df)
    df = encode_target(df)
    X, y = get_X_y(df)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
