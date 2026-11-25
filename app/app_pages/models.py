import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

import common as C
from common import P

C.page_header(
    "Đánh giá mô hình",
    ":material/model_training:",
    "So sánh 10 thuật toán theo quy trình không rò rỉ dữ liệu: CV 5-fold (SMOTE trong từng fold) "
    "để chọn ngưỡng và mô hình, tập test độc lập để kiểm chứng.",
)
C.require_models()

cmp_df = C.get_comparison()
if cmp_df is None or "test_f1_macro" not in cmp_df.columns:
    st.warning("Bảng so sánh đang ở định dạng cũ. Chạy lại `python run_pipeline.py`.",
               icon=":material/warning:")
    st.stop()

cfg = C.get_config()
best = C.best_model_name()
RECALL_T = float(cfg.get("recall_target", 0.75))
F1_T = 0.70
GRID = "rgba(128,128,128,0.15)"

with st.expander("Vì sao số liệu thấp hơn phiên bản trước?", icon=":material/help:"):
    st.markdown(
        "- **Bản cũ** chạy cross-validation trên dữ liệu *đã SMOTE*: mẫu tổng hợp sinh ra từ cùng một "
        "sinh viên rơi vào cả phần train lẫn phần kiểm tra của fold → F1 ≈ 0.87, nhưng trên tập test "
        "thật chỉ ≈ 0.69.\n"
        "- **Bản mới** đặt scaler + SMOTE *bên trong* từng fold, nên CV phản ánh đúng dữ liệu thực.\n"
        "- **Ngưỡng cảnh báo** trước đây dùng chung 28% cho mọi mô hình (AdaBoost cảnh báo 100% sinh "
        "viên). Giờ mỗi mô hình có ngưỡng riêng, chọn trên dự đoán out-of-fold — không dùng tập test.\n"
        "- **Mô hình tốt nhất** được chọn chỉ dựa trên CV; tập test chỉ dùng để báo cáo."
    )

# ── Bảng xếp hạng ──────────────────────────────────────────────────────────
st.subheader("Bảng xếp hạng", icon=":material/leaderboard:")
lb = cmp_df.copy()
lb["Đạt mục tiêu"] = (lb["test_recall_dropout"] >= RECALL_T) & (lb["test_f1_macro"] >= F1_T)
lb.insert(0, "Mô hình", np.where(lb["Model"] == best, lb["Model"] + " ★", lb["Model"]))
lb = lb[["Mô hình", "threshold", "f1_macro", "recall_dropout", "auc_macro", "std_f1",
         "test_f1_macro", "test_recall_dropout", "test_precision_dropout", "test_auc_macro",
         "Đạt mục tiêu"]]
pc = lambda label, help_=None: st.column_config.ProgressColumn(  # noqa: E731
    label, format="%.3f", min_value=0, max_value=1, help=help_)
st.dataframe(
    lb, hide_index=True, alt="Bảng xếp hạng mô hình theo CV và test",
    column_config={
        "threshold": st.column_config.NumberColumn("Ngưỡng", format="percent",
                                                   help="Ngưỡng P(Bỏ học) riêng của mô hình"),
        "f1_macro": pc("CV F1-macro", "Trung bình out-of-fold trên tập train (chọn mô hình theo cột này)"),
        "recall_dropout": pc("CV Recall(BH)"),
        "auc_macro": pc("CV AUC"),
        "std_f1": st.column_config.NumberColumn("Độ lệch F1", format="%.3f",
                                                help="Độ lệch chuẩn F1 giữa 5 fold — càng nhỏ càng ổn định"),
        "test_f1_macro": pc("Test F1-macro"),
        "test_recall_dropout": pc("Test Recall(BH)", "Tỷ lệ SV bỏ học bị phát hiện trên tập test"),
        "test_precision_dropout": pc("Test Precision(BH)", "Trong số SV bị cảnh báo, bao nhiêu % thực sự bỏ học"),
        "test_auc_macro": pc("Test AUC"),
        "Đạt mục tiêu": st.column_config.CheckboxColumn(
            help=f"Test Recall(BH) ≥ {RECALL_T:.2f} và Test F1-macro ≥ {F1_T:.2f}"),
    },
)
st.caption("★ = mô hình tốt nhất theo CV (đặt mặc định). BH = bỏ học.")


def dumbbell(metric: str, title: str, target: float) -> go.Figure:
    d = cmp_df.sort_values(metric)
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_shape(type="line", x0=r[metric], x1=r[f"test_{metric}"], y0=r["Model"], y1=r["Model"],
                      line=dict(color=C.NEUTRAL, width=2))
    fig.add_trace(go.Scatter(x=d[metric], y=d["Model"], mode="markers", name="CV (out-of-fold)",
                             marker=dict(size=11, color=C.SERIES[0], line=dict(width=2, color="white")),
                             hovertemplate="%{y} · CV: %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d[f"test_{metric}"], y=d["Model"], mode="markers", name="Tập test",
                             marker=dict(size=11, color=C.SERIES[1], symbol="diamond",
                                         line=dict(width=2, color="white")),
                             hovertemplate="%{y} · Test: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=target, line_dash="dash", line_color="#6250d6", line_width=1.5,
                  annotation_text=f"Mục tiêu {target:.2f}", annotation_position="bottom right")
    C.style_fig(fig, height=400, legend=True, title=dict(text=title, font=dict(size=14), y=0.98))
    fig.update_layout(margin=dict(t=60))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig


c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    C.show_chart(dumbbell("f1_macro", "F1-macro: CV so với test", F1_T),
                 alt="So sánh F1-macro giữa CV và test của từng mô hình")
with c2, st.container(border=True):
    C.show_chart(dumbbell("recall_dropout", "Recall (bỏ học): CV so với test", RECALL_T),
                 alt="So sánh Recall bỏ học giữa CV và test của từng mô hình")
st.caption("Khoảng cách ngắn giữa hai điểm = mô hình tổng quát hoá tốt, số liệu CV đáng tin.")

# ── Phân tích chi tiết 1 mô hình ───────────────────────────────────────────
st.subheader("Phân tích chi tiết trên tập test", icon=":material/biotech:")
X_test, y_test = C.get_test_data()
if X_test is None:
    st.info("Không tìm thấy data/processed/X_test.csv", icon=":material/info:")
    st.stop()

avail = C.available_models()
name = st.selectbox("Mô hình", avail, index=avail.index(C.active_model()), key="mdl_detail",
                    format_func=lambda m: f"{m} ★" if m == best else m)


@st.fragment
def detail(name: str) -> None:
    proba = C.test_proba(name)
    t_saved = C.threshold_for(name)
    t = st.slider("Thử ngưỡng cảnh báo khác", 0.05, 0.95, float(t_saved), 0.01, key=f"thr_{name}",
                  format="%.2f",
                  help="Chỉ để khám phá đánh đổi Recall/Precision. Ngưỡng chính thức được chọn trên CV.")
    pred = np.where(proba[:, 0] >= t, 0, 1 + np.argmax(proba[:, 1:], axis=1))
    is_d = y_test == 0
    flag = proba[:, 0] >= t
    tp, fn, fp = int((is_d & flag).sum()), int((is_d & ~flag).sum()), int((~is_d & flag).sum())
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    f1m = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)

    def at(th):
        f = proba[:, 0] >= th
        p_ = np.where(f, 0, 1 + np.argmax(proba[:, 1:], axis=1))
        tp_, fn_, fp_ = (is_d & f).sum(), (is_d & ~f).sum(), (~is_d & f).sum()
        return tp_ / max(tp_ + fn_, 1), tp_ / max(tp_ + fp_, 1), f1_score(y_test, p_, average="macro"), \
            accuracy_score(y_test, p_)

    r0, p0, f0, a0 = at(t_saved)
    changed = abs(t - t_saved) > 1e-9
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall (bỏ học)", C.pct(rec), delta=f"{(rec - r0) * 100:+.1f} đ%" if changed else None,
              border=True, help=f"Bắt được {tp}/{tp + fn} SV bỏ học")
    m2.metric("Precision (bỏ học)", C.pct(prec), delta=f"{(prec - p0) * 100:+.1f} đ%" if changed else None,
              border=True, help=f"{tp}/{tp + fp} cảnh báo là đúng")
    m3.metric("F1-macro", f"{f1m:.3f}", delta=f"{f1m - f0:+.3f}" if changed else None, border=True)
    m4.metric("Accuracy", C.pct(acc), delta=f"{(acc - a0) * 100:+.1f} đ%" if changed else None, border=True)

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        labels = [C.CLASS_VI[c] for c in P.CLASS_NAMES]
        fig = go.Figure(go.Heatmap(
            z=cm_pct, x=labels, y=labels, colorscale=[[0, "rgba(42,120,214,0.04)"], [1, "#1c5cab"]],
            zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
            text=[[f"{cm[i, j]}<br>{cm_pct[i, j]:.0%}" for j in range(3)] for i in range(3)],
            texttemplate="%{text}", textfont=dict(size=14),
            hovertemplate="Thực tế %{y} → dự đoán %{x}<br>%{text}<extra></extra>",
        ))
        C.style_fig(fig, height=330, title=dict(text=f"Ma trận nhầm lẫn · ngưỡng {t:.0%}", font=dict(size=14)))
        fig.update_xaxes(title="Dự đoán", side="bottom")
        fig.update_yaxes(title="Thực tế", autorange="reversed")
        C.show_chart(fig, alt="Ma trận nhầm lẫn 3 lớp trên tập test")

    with b, st.container(border=True):
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
        curves = np.array([at(g)[:3] for g in grid])
        fig = go.Figure()
        for i, (nm, col) in enumerate([("Recall (bỏ học)", C.SERIES[1]), ("Precision (bỏ học)", C.SERIES[0]),
                                       ("F1-macro", C.SERIES[2])]):
            fig.add_trace(go.Scatter(x=grid, y=curves[:, i], mode="lines", name=nm,
                                     line=dict(color=col, width=2),
                                     hovertemplate=f"{nm}: %{{y:.3f}}<extra></extra>"))
        fig.add_vline(x=t_saved, line_dash="dash", line_color="#6250d6", line_width=1.5,
                      annotation_text=f"Đã chọn {t_saved:.0%}", annotation_position="top left")
        if changed:
            fig.add_vline(x=t, line_color=C.NEUTRAL, line_width=1.5)
        C.style_fig(fig, height=330, legend=True, hovermode="x unified",
                    title=dict(text="Đánh đổi theo ngưỡng", font=dict(size=14), y=0.98))
        fig.update_layout(margin=dict(t=60))
        fig.update_xaxes(tickformat=".0%", title="Ngưỡng P(Bỏ học)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(range=[0, 1.02], showgrid=True, gridcolor=GRID)
        C.show_chart(fig, alt="Recall, Precision, F1 theo ngưỡng")

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        fpr, tpr, _ = roc_curve(is_d.astype(int), proba[:, 0])
        auc = roc_auc_score(is_d.astype(int), proba[:, 0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=C.NEUTRAL, width=1),
                                 hoverinfo="skip", name="Ngẫu nhiên"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=C.SERIES[0], width=2),
                                 fill="tozeroy", fillcolor=C.hex_rgba(C.SERIES[0], 0.12),
                                 name=f"AUC = {auc:.3f}",
                                 hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
        fpr_t = fp / max(int((~is_d).sum()), 1)
        fig.add_trace(go.Scatter(x=[fpr_t], y=[rec], mode="markers", name=f"Ngưỡng {t:.0%}",
                                 marker=dict(size=12, color="#6250d6", line=dict(width=2, color="white"))))
        C.style_fig(fig, height=330, legend=True,
                    title=dict(text="ROC — Bỏ học so với còn lại", font=dict(size=14), y=0.98))
        fig.update_layout(margin=dict(t=60))
        fig.update_xaxes(range=[0, 1], title="Tỷ lệ báo nhầm (FPR)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(range=[0, 1.02], title="Tỷ lệ phát hiện (TPR)", showgrid=True, gridcolor=GRID)
        C.show_chart(fig, alt="Đường cong ROC cho lớp bỏ học")

    with b, st.container(border=True):
        model = C.get_model(name)
        imp = None
        if hasattr(model, "booster_"):  # LightGBM: mặc định đếm số lần split → dùng gain
            imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=C.FEATURE_COLS)
            note = "Độ quan trọng theo tổng gain của mô hình."
        elif hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=C.FEATURE_COLS)
            note = "Độ quan trọng (giảm tạp chất / gain) của mô hình."
        elif hasattr(model, "coef_"):
            imp = pd.Series(np.abs(model.coef_[0]), index=C.FEATURE_COLS)
            note = "|Hệ số| của lớp bỏ học (dữ liệu đã chuẩn hoá)."
        if imp is not None:
            imp = (imp / imp.sum()).sort_values().tail(12)
            fig = go.Figure(go.Bar(x=imp.values * 100, y=[C.feature_vi(c) for c in imp.index], orientation="h",
                                   marker=dict(color=C.SERIES[0], cornerradius=4),
                                   hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            C.style_fig(fig, height=330, title=dict(text="12 đặc trưng quan trọng nhất", font=dict(size=14)))
            fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID)
            C.show_chart(fig, alt="Độ quan trọng đặc trưng của mô hình")
            st.caption(note)
        else:
            st.markdown("**Độ quan trọng đặc trưng**")
            st.caption(f"{name} không cung cấp độ quan trọng đặc trưng trực tiếp. "
                       "Xem phần *Yếu tố ảnh hưởng* ở trang Dự đoán cá nhân để có giải thích theo từng hồ sơ.")


detail(name)
