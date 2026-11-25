"""
app.py — Web app cảnh báo sớm nguy cơ bỏ học của sinh viên
==========================================================
Chạy: streamlit run app/app.py
"""

from datetime import datetime

import streamlit as st

import common as C

st.set_page_config(
    page_title="EduRisk · Dự đoán nguy cơ bỏ học",
    page_icon=C.ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

C.init_state()
st.logo(C.LOGO_PATH, icon_image=C.ICON_PATH, size="large")

PAGES = {
    "": [
        st.Page("app_pages/overview.py", title="Tổng quan", icon=":material/space_dashboard:",
                default=True),
    ],
    "Dự đoán": [
        st.Page("app_pages/predict.py", title="Dự đoán cá nhân", icon=":material/person_search:"),
        st.Page("app_pages/batch.py", title="Dự đoán hàng loạt", icon=":material/groups:"),
    ],
    "Phân tích": [
        st.Page("app_pages/eda.py", title="Khám phá dữ liệu", icon=":material/insights:"),
        st.Page("app_pages/models.py", title="Đánh giá mô hình", icon=":material/model_training:"),
    ],
}
page = st.navigation(PAGES, position="sidebar")

# ── Sidebar: chọn mô hình dùng chung + trạng thái ────────────────────────────
with st.sidebar:
    if C.model_ready():
        avail = C.available_models()
        best = C.best_model_name()
        st.session_state.setdefault("active_model", best)
        if st.session_state["active_model"] not in avail:
            st.session_state["active_model"] = best
        st.selectbox(
            "Mô hình đang dùng",
            options=avail,
            key="active_model",
            format_func=lambda m: f"{m}  ★" if m == best else m,
            help="Áp dụng cho dự đoán cá nhân và hàng loạt. ★ = mô hình tốt nhất theo CV.",
        )
        cfg = C.get_config()
        t = C.threshold_for(st.session_state["active_model"])
        with st.container(border=True, gap="xsmall"):
            st.markdown(f":material/tune: Ngưỡng cảnh báo **{t:.0%}**")
            if cfg.get("trained_at"):
                trained = datetime.fromisoformat(cfg["trained_at"]).strftime("%d/%m/%Y %H:%M")
                st.caption(f"Huấn luyện lúc {trained} · {cfg.get('n_train', '?')} SV train, "
                           f"{cfg.get('n_test', '?')} SV test")
        n_hist = len(st.session_state["history"])
        if n_hist:
            st.caption(f":material/history: {n_hist} lượt dự đoán trong phiên này")
    else:
        st.warning("Chưa có mô hình. Chạy `python run_pipeline.py`.", icon=":material/warning:")
    st.caption("Dữ liệu: UCI · Predict Students' Dropout and Academic Success (4.424 SV)")

page.run()
