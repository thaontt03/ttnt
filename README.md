# 🎓 Dự đoán nguy cơ bỏ học sinh viên

> **Môn học:** Trí tuệ Nhân tạo — Đại học Công nghiệp Hà Nội  
> **Dataset:** [UCI — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697)  
> **Ngôn ngữ:** Python 3.10+

---

## 📋 Mô tả dự án

Dự án xây dựng mô hình Machine Learning để **dự đoán nguy cơ bỏ học của sinh viên** dựa trên dữ liệu học tập, nhân khẩu học và kinh tế - xã hội.

**Bài toán:** Phân loại đa lớp — Dropout / Enrolled / Graduate  
**Metric chính:** Recall(Dropout) ≥ 0.75, F1-macro ≥ 0.70

---

## 🗂️ Cấu trúc dự án

```
dropout-prediction/
├── data/
│   ├── raw/dataset.csv              # Dataset gốc từ UCI (cần tải thủ công)
│   └── processed/                   # Dữ liệu sau xử lý
├── notebooks/
│   ├── 01_EDA_basic.ipynb
│   ├── 02_EDA_visual.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_Interpretation.ipynb
├── src/
│   ├── data/load_data.py            # Đọc và kiểm tra dataset
│   ├── data/preprocess.py           # Pipeline: scaling, SMOTE
│   ├── data/codebook.py             # Bảng mã tiếng Việt cho biến phân loại
│   ├── models/train.py              # 10 thuật toán · CV chuẩn · chọn ngưỡng
│   ├── models/evaluate.py           # Đánh giá: CM, ROC, metrics
│   ├── models/predict.py            # Dự đoán + giải thích Shapley theo nhóm
│   └── visualization/plots.py       # Hàm vẽ biểu đồ (notebooks)
├── models/                          # .pkl + model_comparison.csv + predict_config.json
├── app/
│   ├── app.py                       # Điểm vào Streamlit (điều hướng, sidebar)
│   ├── common.py                    # Loader có cache, biểu đồ, form, khuyến nghị
│   ├── app_pages/                   # overview · predict · batch · eda · models
│   ├── assets/                      # Logo
│   └── .streamlit/config.toml       # Theme sáng/tối, font Be Vietnam Pro
├── run_pipeline.py                  # Huấn luyện toàn bộ (khuyến nghị)
├── reports/figures/                 # Ảnh biểu đồ
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/<ten-nhom>/dropout-prediction.git
cd dropout-prediction
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

### 4. Tải dataset

Tải dataset từ [UCI ML Repository](https://archive.ics.uci.edu/dataset/697) và đặt vào:
```
data/raw/dataset.csv
```
> File CSV dùng dấu **chấm phẩy (;)** làm separator.

---

## 🚀 Chạy dự án

### Bước 1 (khuyến nghị): Huấn luyện bằng script

```bash
python run_pipeline.py                         # 10 mô hình, ~1.5 phút
python run_pipeline.py --models XGBoost LightGBM --cv 3
```

Script tạo toàn bộ file trong `models/` và `data/processed/` theo quy trình **không rò rỉ dữ liệu**:

1. Tách train/test stratified (80/20, `random_state=42`).
2. Mỗi mô hình: CV 5-fold với **scaler + SMOTE nằm trong từng fold** → dự đoán out-of-fold.
3. Chọn **ngưỡng P(Dropout) riêng cho từng mô hình** trên OOF: F1-macro cao nhất trong các ngưỡng có
   Recall(Dropout) ≥ 0.78 (mục tiêu 0.75 + biên an toàn 0.03).
4. Chọn mô hình tốt nhất **chỉ theo CV**; tập test dùng đúng 1 lần để báo cáo.

### Bước 1 (tuỳ chọn): Chạy các notebooks theo thứ tự

```bash
jupyter notebook
```

Chạy lần lượt từ `01` đến `06`:
1. `01_EDA_basic.ipynb` — Thống kê mô tả
2. `02_EDA_visual.ipynb` — Trực quan hóa nâng cao
3. `03_Preprocessing.ipynb` — Tiền xử lý, SMOTE → tạo file trong `data/processed/` và `models/preprocessing_pipeline.pkl`
4. `04_Modeling.ipynb` — Train 7 mô hình → tạo `models/best_model.pkl`
5. `05_Evaluation.ipynb` — Đánh giá mô hình
6. `06_Interpretation.ipynb` — SHAP analysis

### Bước 2: Chạy web app

```bash
streamlit run app/app.py
```

Mở trình duyệt tại: `http://localhost:8501`

---

## 🌐 Web App — Các trang

| Trang | Mô tả |
|-------|-------|
| Tổng quan | KPI, cách hệ thống ra quyết định, lối tắt |
| Dự đoán cá nhân | Form đủ 36 thuộc tính (tiếng Việt), hồ sơ mẫu, so sánh nhiều mô hình, đồng hồ rủi ro, giải thích Shapley, mô phỏng *what-if*, khuyến nghị, lịch sử |
| Dự đoán hàng loạt | Upload CSV (tự nhận `,` `;` tab), lọc theo mức rủi ro/ngành, đối chiếu với cột `Target`, xuất CSV |
| Khám phá dữ liệu | Tỷ lệ bỏ học theo tài chính, học lực, ngành, nhân khẩu, tương quan — có bộ lọc |
| Đánh giá mô hình | Bảng xếp hạng CV + test, ma trận nhầm lẫn, ROC, thử ngưỡng tương tác, độ quan trọng đặc trưng |

**Quy ước kết luận:** *Có nguy cơ bỏ học* ⇔ P(Dropout) ≥ ngưỡng của mô hình. Mức rủi ro tính theo ngưỡng `t`:
Thấp (< 0.7t) · Cần theo dõi (0.7t – t) · Cao (t – (1+t)/2) · Rất cao (≥ (1+t)/2).

---

## 🤖 Thuật toán ML & kết quả

10 thuật toán: Logistic Regression, Decision Tree, Random Forest, Extra Trees, SVM (RBF), KNN,
Gradient Boosting, AdaBoost, XGBoost, LightGBM. Kết quả đầy đủ: `models/model_comparison.csv`
hoặc trang **Đánh giá mô hình**.

Mô hình được chọn (theo CV): **LightGBM**, ngưỡng 0.45 — trên tập test (885 SV):
Recall(Dropout) **0.750**, Precision(Dropout) 0.798, F1-macro **0.699**, AUC 0.889.

> Phiên bản trước báo cáo F1 ≈ 0.87 vì cross-validation chạy trên dữ liệu đã SMOTE (rò rỉ
> mẫu tổng hợp giữa các fold). Số liệu hiện tại là số trung thực.

---

## 🔧 Yêu cầu hệ thống

- Python 3.10+ (đã kiểm thử 3.12, Streamlit 1.66)
- RAM: 4GB+
- Disk: 500MB (sau khi cài thư viện)

---

## 📚 Tài liệu tham khảo

1. Realinho et al. (2022). Predict Students' Dropout and Academic Success. UCI ML Repository.
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
3. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
4. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
5. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions.
