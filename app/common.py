"""
common.py — Tiện ích dùng chung cho các trang của web app.

- Loader có cache (tự làm mới khi file model/dữ liệu thay đổi nhờ khoá theo mtime)
- Bảng màu, badge mức rủi ro, biểu đồ Plotly dùng lại
- Đặc tả form nhập liệu sinh viên + khuyến nghị can thiệp
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib  # noqa: E402

from src.data.codebook import CODEBOOK, FEATURE_VI, feature_vi, label  # noqa: E402,F401
from src.data.load_data import FEATURE_COLS  # noqa: E402
from src.models import predict as P  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# ĐƯỜNG DẪN
# ─────────────────────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "raw", "dataset.csv")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "predict_config.json")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
X_TEST_PATH = os.path.join(ROOT, "data", "processed", "X_test.csv")
Y_TEST_PATH = os.path.join(ROOT, "data", "processed", "y_test.csv")
LOGO_PATH = os.path.join(APP_DIR, "assets", "logo.svg")
ICON_PATH = os.path.join(APP_DIR, "assets", "icon.svg")

# ─────────────────────────────────────────────────────────────────────────────
# MÀU SẮC (status palette cố định cho mức rủi ro; categorical cho lớp)
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLOR = {"Thấp": "#0ca30c", "Cần theo dõi": "#fab219", "Cao": "#ec835a", "Rất cao": "#d03b3b"}
RISK_BADGE = {
    "Thấp": ("green", ":material/check_circle:"),
    "Cần theo dõi": ("yellow", ":material/visibility:"),
    "Cao": ("orange", ":material/warning:"),
    "Rất cao": ("red", ":material/emergency_home:"),
}
RISK_DESC = {
    "Thấp": "Xác suất bỏ học thấp hơn nhiều so với ngưỡng cảnh báo.",
    "Cần theo dõi": "Chưa vượt ngưỡng nhưng đã tiến gần — nên theo dõi định kỳ.",
    "Cao": "Vượt ngưỡng cảnh báo — nên liên hệ, tư vấn sớm.",
    "Rất cao": "Vượt xa ngưỡng cảnh báo — cần can thiệp ưu tiên.",
}
CLASS_COLOR = {"Dropout": "#e34948", "Enrolled": "#eda100", "Graduate": "#2a78d6"}
CLASS_VI = {"Dropout": "Bỏ học", "Enrolled": "Đang học", "Graduate": "Tốt nghiệp"}
BIN_COLOR = {P.LABEL_AT_RISK: "#d03b3b", P.LABEL_SAFE: "#2a78d6"}
INCREASE, DECREASE = "#d03b3b", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#9aa0b8"
PLOT_CONFIG = {"displaylogo": False,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toImage"]}


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# LOADER CÓ CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(show_spinner="Đang tải dữ liệu…")
def _load_dataset(_mt: float) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    df["Dropout"] = (df["Target"] == "Dropout").astype(int)
    df["Kết quả"] = df["Target"].map(CLASS_VI)
    df["Nguy cơ"] = np.where(df["Dropout"] == 1, P.LABEL_AT_RISK, P.LABEL_SAFE)
    return df


def get_dataset() -> pd.DataFrame | None:
    return _load_dataset(_mtime(DATA_PATH)) if os.path.exists(DATA_PATH) else None


@st.cache_data(show_spinner=False)
def _load_config(_mt: float) -> dict:
    return P.load_config(CONFIG_PATH)


def get_config() -> dict:
    return _load_config(_mtime(CONFIG_PATH))


@st.cache_resource(show_spinner="Đang tải pipeline…")
def _load_pipeline(_mt: float):
    return joblib.load(PIPELINE_PATH)


def get_pipeline():
    return _load_pipeline(_mtime(PIPELINE_PATH)) if os.path.exists(PIPELINE_PATH) else None


@st.cache_resource(show_spinner="Đang tải mô hình…", max_entries=12)
def _load_model(name: str, _mt: float):
    return joblib.load(os.path.join(MODELS_DIR, P.model_filename(name)))


def get_model(name: str):
    return _load_model(name, _mtime(os.path.join(MODELS_DIR, P.model_filename(name))))


def available_models() -> list[str]:
    return P.list_available_models(MODELS_DIR)


def best_model_name() -> str | None:
    avail = available_models()
    best = get_config().get("best_model")
    return best if best in avail else (avail[0] if avail else None)


def active_model() -> str | None:
    """Mô hình đang chọn ở sidebar (mặc định: mô hình tốt nhất)."""
    m = st.session_state.get("active_model")
    return m if m in available_models() else best_model_name()


def threshold_for(name: str | None) -> float:
    return P.get_threshold(get_config(), name)


def model_ready() -> bool:
    return bool(available_models()) and os.path.exists(PIPELINE_PATH)


@st.cache_data(show_spinner=False)
def _load_comparison(_mt: float) -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH)


def get_comparison() -> pd.DataFrame | None:
    return _load_comparison(_mtime(COMPARISON_PATH)) if os.path.exists(COMPARISON_PATH) else None


@st.cache_data(show_spinner=False)
def _load_test(_mx: float, _my: float):
    X = pd.read_csv(X_TEST_PATH).values
    y = pd.read_csv(Y_TEST_PATH).squeeze().values.astype(int)
    return X, y


def get_test_data():
    if not (os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH)):
        return None, None
    return _load_test(_mtime(X_TEST_PATH), _mtime(Y_TEST_PATH))


@st.cache_data(show_spinner="Đang chấm điểm tập test…", max_entries=12)
def _test_proba(name: str, _mm: float, _mx: float) -> np.ndarray:
    X, _ = get_test_data()
    return get_model(name).predict_proba(X)


def test_proba(name: str) -> np.ndarray:
    return _test_proba(name, _mtime(os.path.join(MODELS_DIR, P.model_filename(name))),
                       _mtime(X_TEST_PATH))


@st.cache_data(show_spinner=False)
def _test_index(_mt: float, test_size: float) -> pd.Index:
    """Tái tạo đúng phép chia train/test lúc huấn luyện (stratified, random_state=42)."""
    from src.data.preprocess import split_raw
    df = _load_dataset(_mt)
    _, X_te, _, _ = split_raw(df[FEATURE_COLS], df["Target"], test_size=test_size)
    return X_te.index


def test_index() -> pd.Index:
    return _test_index(_mtime(DATA_PATH), float(get_config().get("test_size", 0.2)))


CATEGORICAL = set(CODEBOOK) | {"Application order"}


@st.cache_data(show_spinner=False)
def _reference_profile(_mt: float) -> pd.Series:
    """Hồ sơ 'điển hình': trung vị cho biến số, mode cho biến phân loại."""
    df = _load_dataset(_mt)
    ref = {}
    for c in FEATURE_COLS:
        ref[c] = float(df[c].mode().iloc[0]) if c in CATEGORICAL else float(df[c].median())
    return pd.Series(ref)


def reference_profile() -> pd.Series:
    return _reference_profile(_mtime(DATA_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# THÀNH PHẦN GIAO DIỆN
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title: str, icon: str, caption: str) -> None:
    st.title(title, icon=icon)
    st.caption(caption)


def require_models() -> None:
    if not model_ready():
        st.error("Chưa có mô hình đã huấn luyện. Chạy `python run_pipeline.py` ở thư mục gốc dự án "
                 "rồi tải lại trang.", title="Thiếu mô hình", icon=":material/error:")
        st.stop()


def risk_badge(level: str) -> None:
    color, icon = RISK_BADGE[level]
    st.badge(f"Mức rủi ro: {level}", icon=icon, color=color)


def risk_md(level: str) -> str:
    color, _ = RISK_BADGE[level]
    return f":{color}-badge[{level}]"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def style_fig(fig: go.Figure, height: int = 320, legend: bool = False, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30 if layout.get("title") else 10, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_size=13),
        **{"bargap": 0.3, **layout},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def show_chart(fig: go.Figure, alt: str, key: str | None = None) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, alt=alt, key=key)


def gauge_figure(p: float, t: float, height: int = 230) -> go.Figure:
    """Đồng hồ P(Dropout) với 4 vùng rủi ro và vạch ngưỡng."""
    level = P.risk_level(p, t)
    bands = [(0, 0.7 * t, "Thấp"), (0.7 * t, t, "Cần theo dõi"),
             (t, (1 + t) / 2, "Cao"), ((1 + t) / 2, 1, "Rất cao")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number=dict(suffix="%", valueformat=".1f", font=dict(size=44, color=RISK_COLOR[level])),
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, 100], ticksuffix="%", tickwidth=1, nticks=6),
            bar=dict(color=RISK_COLOR[level], thickness=0.32),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[dict(range=[a * 100, b * 100], color=hex_rgba(RISK_COLOR[n], 0.22))
                   for a, b, n in bands],
            threshold=dict(line=dict(color="#6250d6", width=4), thickness=0.9, value=t * 100),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(height=height, margin=dict(l=36, r=44, t=20, b=0))
    return fig


def proba_bar(probabilities: dict, height: int = 92) -> go.Figure:
    """Thanh 100% xác suất 3 lớp."""
    fig = go.Figure()
    for cls in ["Dropout", "Enrolled", "Graduate"]:
        v = probabilities[cls]
        fig.add_trace(go.Bar(
            x=[v * 100], y=[""], orientation="h", name=CLASS_VI[cls],
            marker=dict(color=CLASS_COLOR[cls], line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[f"{CLASS_VI[cls]} {v * 100:.0f}%" if v >= 0.12 else ""],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=13),
            hovertemplate=f"{CLASS_VI[cls]}: %{{x:.1f}}%<extra></extra>",
        ))
    style_fig(fig, height=height, legend=True, barmode="stack")
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0), legend=dict(y=1.08, traceorder="normal"))
    return fig


def impact_figure(expl: pd.DataFrame, top: int = 8, height: int | None = None) -> go.Figure:
    """Biểu đồ thanh hai chiều: nhóm yếu tố làm tăng (đỏ) / giảm (xanh) nguy cơ."""
    d = expl.head(top).iloc[::-1]
    colors = [INCREASE if v > 0 else DECREASE for v in d["impact"]]
    fig = go.Figure(go.Bar(
        x=d["impact"] * 100, y=d["group"], orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v * 100:+.1f} đ%" for v in d["impact"]],
        textposition="outside", cliponaxis=False,
        customdata=d["p_if_typical"] * 100,
        hovertemplate="<b>%{y}</b><br>Đóng góp (Shapley): %{x:+.1f} điểm %"
                      "<br>Nếu riêng nhóm này về mức điển hình: P = %{customdata:.1f}%<extra></extra>",
    ))
    lim = max(5.0, float(np.abs(d["impact"]).max() * 100) * 1.35)
    style_fig(fig, height=height or 44 * len(d) + 40)
    fig.update_xaxes(range=[-lim, lim], ticksuffix=" đ%", showgrid=True,
                     gridcolor="rgba(128,128,128,0.15)")
    fig.add_vline(x=0, line_width=1, line_color=NEUTRAL)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ĐẶC TẢ FORM NHẬP LIỆU
# ─────────────────────────────────────────────────────────────────────────────

def wkey(feature: str) -> str:
    return "in__" + "".join(ch if ch.isalnum() else "_" for ch in feature)


# (feature, kiểu, tham số) — kiểu: int | float | select | bool
FORM_SPEC = {
    "Học tập": {
        "Học kỳ 1": [
            ("Curricular units 1st sem (enrolled)", "int", dict(min=0, max=26, help="Số môn đã đăng ký học trong học kỳ 1")),
            ("Curricular units 1st sem (approved)", "int", dict(min=0, max=26, help="Số môn qua (đạt) trong học kỳ 1")),
            ("Curricular units 1st sem (grade)", "float", dict(min=0.0, max=20.0, step=0.1, help="Điểm TB thang 20 — để 0 nếu không đạt môn nào")),
            ("Curricular units 1st sem (evaluations)", "int", dict(min=0, max=45, help="Tổng số lượt dự thi/đánh giá")),
            ("Curricular units 1st sem (credited)", "int", dict(min=0, max=20, help="Số môn được công nhận/miễn")),
            ("Curricular units 1st sem (without evaluations)", "int", dict(min=0, max=12, help="Số môn đăng ký nhưng không dự thi")),
        ],
        "Học kỳ 2": [
            ("Curricular units 2nd sem (enrolled)", "int", dict(min=0, max=23, help="Số môn đã đăng ký học trong học kỳ 2")),
            ("Curricular units 2nd sem (approved)", "int", dict(min=0, max=23, help="Số môn qua (đạt) trong học kỳ 2")),
            ("Curricular units 2nd sem (grade)", "float", dict(min=0.0, max=20.0, step=0.1, help="Điểm TB thang 20 — để 0 nếu không đạt môn nào")),
            ("Curricular units 2nd sem (evaluations)", "int", dict(min=0, max=33, help="Tổng số lượt dự thi/đánh giá")),
            ("Curricular units 2nd sem (credited)", "int", dict(min=0, max=19, help="Số môn được công nhận/miễn")),
            ("Curricular units 2nd sem (without evaluations)", "int", dict(min=0, max=12, help="Số môn đăng ký nhưng không dự thi")),
        ],
    },
    "Tài chính": {
        "": [
            ("Tuition fees up to date", "bool", dict(help="Sinh viên đã đóng học phí đúng hạn")),
            ("Debtor", "bool", dict(help="Sinh viên đang có khoản nợ với trường")),
            ("Scholarship holder", "bool", dict(help="Sinh viên đang nhận học bổng")),
        ],
    },
    "Cá nhân": {
        "": [
            ("Age at enrollment", "int", dict(min=17, max=70, help="Tuổi tại thời điểm nhập học")),
            ("Gender", "select", dict()),
            ("Marital status", "select", dict()),
            ("Nacionality", "select", dict()),
            ("Displaced", "bool", dict(help="Sinh viên phải chuyển nơi ở để đi học")),
            ("International", "bool", dict()),
            ("Educational special needs", "bool", dict()),
        ],
    },
    "Tuyển sinh": {
        "": [
            ("Course", "select", dict()),
            ("Daytime/evening attendance", "select", dict()),
            ("Application mode", "select", dict()),
            ("Application order", "int", dict(min=0, max=9, help="0 = nguyện vọng 1 … 9 = nguyện vọng cuối")),
            ("Previous qualification", "select", dict()),
            ("Previous qualification (grade)", "float", dict(min=0.0, max=200.0, step=0.1, help="Thang 0–200")),
            ("Admission grade", "float", dict(min=0.0, max=200.0, step=0.1, help="Thang 0–200")),
        ],
    },
    "Gia đình": {
        "": [
            ("Mother's qualification", "select", dict()),
            ("Father's qualification", "select", dict()),
            ("Mother's occupation", "select", dict()),
            ("Father's occupation", "select", dict()),
        ],
    },
}

MACRO_COLS = ["Unemployment rate", "Inflation rate", "GDP"]
MACRO_KEY = "in__macro"


@st.cache_data(show_spinner=False)
def _select_options(_mt: float) -> dict[str, list[int]]:
    df = _load_dataset(_mt)
    return {f: sorted(int(v) for v in df[f].unique()) for f in CODEBOOK}


def select_options(feature: str) -> list[int]:
    return _select_options(_mtime(DATA_PATH))[feature]


@st.cache_data(show_spinner=False)
def _macro_options(_mt: float) -> list[tuple[float, float, float]]:
    df = _load_dataset(_mt)
    return sorted({tuple(float(x) for x in r) for r in df[MACRO_COLS].itertuples(index=False)})


def macro_options() -> list[tuple[float, float, float]]:
    return _macro_options(_mtime(DATA_PATH))


def macro_label(t: tuple) -> str:
    return f"Thất nghiệp {t[0]:.1f}% · Lạm phát {t[1]:.1f}% · GDP {t[2]:+.2f}%"


def all_form_features() -> list[tuple[str, str, dict]]:
    return [item for tab in FORM_SPEC.values() for group in tab.values() for item in group]


def load_student_into_state(student: dict | pd.Series) -> None:
    """Đổ hồ sơ sinh viên vào session_state của các widget form."""
    for feature, kind, _ in all_form_features():
        v = student[feature]
        if kind == "int":
            v = int(round(float(v)))
        elif kind == "float":
            v = float(v)
        elif kind == "bool":
            v = bool(int(v))
        else:
            v = int(v)
            if v not in select_options(feature):
                v = select_options(feature)[0]
        st.session_state[wkey(feature)] = v
    opts = macro_options()
    target = tuple(float(student[c]) for c in MACRO_COLS)
    st.session_state[MACRO_KEY] = min(opts, key=lambda o: sum((a - b) ** 2 for a, b in zip(o, target)))


def student_from_state() -> dict:
    data = {}
    for feature, kind, _ in all_form_features():
        v = st.session_state[wkey(feature)]
        data[feature] = float(int(v)) if kind == "bool" else float(v)
    for c, v in zip(MACRO_COLS, st.session_state[MACRO_KEY]):
        data[c] = float(v)
    return {c: data[c] for c in FEATURE_COLS}


def render_field(feature: str, kind: str, opt: dict) -> None:
    key, name = wkey(feature), feature_vi(feature)
    if kind == "int":
        st.number_input(name, min_value=opt["min"], max_value=opt["max"], step=1,
                        key=key, help=opt.get("help"))
    elif kind == "float":
        st.number_input(name, min_value=opt["min"], max_value=opt["max"], step=opt.get("step", 0.1),
                        format="%.1f", key=key, help=opt.get("help"))
    elif kind == "bool":
        st.toggle(name, key=key, help=opt.get("help"))
    else:
        st.selectbox(name, options=select_options(feature),
                     format_func=lambda c, f=feature: label(f, c), key=key, help=opt.get("help"))


def validate_student(s: dict) -> tuple[list[str], list[str]]:
    """Trả về (lỗi chặn, cảnh báo)."""
    errors, warns = [], []
    for sem, vi in [("1st", "HK1"), ("2nd", "HK2")]:
        e = s[f"Curricular units {sem} sem (enrolled)"]
        a = s[f"Curricular units {sem} sem (approved)"]
        v = s[f"Curricular units {sem} sem (evaluations)"]
        c = s[f"Curricular units {sem} sem (credited)"]
        g = s[f"Curricular units {sem} sem (grade)"]
        if a > e:
            errors.append(f"{vi}: số môn đạt ({a:.0f}) lớn hơn số môn đăng ký ({e:.0f}).")
        if c > e:
            errors.append(f"{vi}: số môn được công nhận ({c:.0f}) lớn hơn số môn đăng ký ({e:.0f}).")
        if a > v:
            warns.append(f"{vi}: số môn đạt ({a:.0f}) lớn hơn số lượt dự thi ({v:.0f}).")
        if a == 0 and g > 0:
            warns.append(f"{vi}: không đạt môn nào nhưng điểm TB = {g:.1f} (dữ liệu gốc luôn là 0).")
        if a > 0 and g == 0:
            warns.append(f"{vi}: có {a:.0f} môn đạt nhưng điểm TB = 0.")
    if s["Debtor"] == 1 and s["Tuition fees up to date"] == 1:
        warns.append("Vừa 'đang nợ' vừa 'đóng học phí đúng hạn' — hãy kiểm tra lại.")
    return errors, warns


# ─────────────────────────────────────────────────────────────────────────────
# KHUYẾN NGHỊ
# ─────────────────────────────────────────────────────────────────────────────

def recommendations(s: dict, result: dict) -> list[tuple[str, str, str]]:
    """Danh sách (icon, tiêu đề, nội dung) dựa trên hồ sơ và kết quả dự đoán."""
    recs = []

    def ratio(sem):
        e = s[f"Curricular units {sem} sem (enrolled)"]
        return s[f"Curricular units {sem} sem (approved)"] / e if e else None

    if s["Tuition fees up to date"] == 0 or s["Debtor"] == 1:
        recs.append((":material/payments:", "Hỗ trợ tài chính",
                     "Học phí chưa hoàn tất/đang nợ là yếu tố rủi ro mạnh nhất trong dữ liệu. "
                     "Tư vấn gia hạn, trả góp hoặc giới thiệu quỹ hỗ trợ."))
    r1, r2 = ratio("1st"), ratio("2nd")
    if (r1 is not None and r1 < 0.6) or (r2 is not None and r2 < 0.6):
        recs.append((":material/school:", "Cố vấn học tập",
                     "Tỷ lệ qua môn dưới 60%. Xếp lịch gặp cố vấn, cân nhắc giảm tải tín chỉ "
                     "và đăng ký lớp phụ đạo."))
    g1, g2 = s["Curricular units 1st sem (grade)"], s["Curricular units 2nd sem (grade)"]
    if 0 < g2 < g1 - 1.5:
        recs.append((":material/trending_down:", "Kết quả đang giảm",
                     f"Điểm TB giảm từ {g1:.1f} (HK1) xuống {g2:.1f} (HK2). Tìm hiểu nguyên nhân sớm."))
    if s["Curricular units 1st sem (without evaluations)"] + s["Curricular units 2nd sem (without evaluations)"] > 0:
        recs.append((":material/event_busy:", "Bỏ thi",
                     "Có môn đăng ký nhưng không dự thi — dấu hiệu sớm của việc rời bỏ. Liên hệ trực tiếp."))
    if s["Curricular units 2nd sem (enrolled)"] == 0 and s["Curricular units 1st sem (enrolled)"] > 0:
        recs.append((":material/person_off:", "Không đăng ký HK2",
                     "Sinh viên không đăng ký môn nào ở học kỳ 2 — cần xác minh tình trạng học tập ngay."))
    if s["Scholarship holder"] == 0 and result["at_risk"]:
        recs.append((":material/workspace_premium:", "Xét học bổng / trợ cấp",
                     "Sinh viên có học bổng có tỷ lệ bỏ học thấp hơn rõ rệt. Cân nhắc các chương trình hỗ trợ."))
    if s["Age at enrollment"] >= 25:
        recs.append((":material/schedule:", "Lịch học linh hoạt",
                     "Sinh viên nhập học lớn tuổi thường vừa học vừa làm — gợi ý lớp tối/trực tuyến."))
    if not recs:
        recs.append((":material/thumb_up:", "Duy trì",
                     "Chưa thấy yếu tố rủi ro nổi bật. Tiếp tục theo dõi theo chu kỳ học kỳ."))
    return recs


def init_state() -> None:
    """Khởi tạo session_state dùng chung (gọi 1 lần ở app.py)."""
    st.session_state.setdefault("history", [])
