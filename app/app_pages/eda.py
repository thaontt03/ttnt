import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C

C.page_header(
    "Khám phá dữ liệu",
    ":material/insights:",
    "Ai có nguy cơ bỏ học cao hơn? Tỷ lệ bỏ học theo từng nhóm sinh viên trong dữ liệu UCI.",
)

df0 = C.get_dataset()
if df0 is None:
    st.error("Không tìm thấy data/raw/dataset.csv", icon=":material/error:")
    st.stop()

GRID = "rgba(128,128,128,0.15)"

# ── Bộ lọc ─────────────────────────────────────────────────────────────────
with st.container(border=True):
    f1, f2, f3, f4 = st.columns([2, 1.1, 1.1, 1.6], vertical_alignment="bottom")
    courses = sorted(df0["Course"].unique())
    sel_course = f1.multiselect("Ngành học", courses, format_func=lambda c: C.label("Course", c),
                                placeholder="Tất cả ngành", key="eda_course")
    sel_gender = f2.segmented_control("Giới tính", [0, 1], format_func=C.CODEBOOK["Gender"].get,
                                      selection_mode="multi", key="eda_gender")
    sel_time = f3.segmented_control("Ca học", [1, 0], format_func=lambda v: "Ngày" if v == 1 else "Tối",
                                    selection_mode="multi", key="eda_time")
    amin, amax = int(df0["Age at enrollment"].min()), int(df0["Age at enrollment"].max())
    sel_age = f4.slider("Tuổi nhập học", amin, amax, (amin, amax), key="eda_age")

df = df0
if sel_course:
    df = df[df["Course"].isin(sel_course)]
if sel_gender:
    df = df[df["Gender"].isin(sel_gender)]
if sel_time:
    df = df[df["Daytime/evening attendance"].isin(sel_time)]
df = df[df["Age at enrollment"].between(*sel_age)]

if df.empty:
    st.warning("Không có sinh viên nào khớp bộ lọc.", icon=":material/filter_alt_off:")
    st.stop()

base_rate = df0["Dropout"].mean()
rate = df["Dropout"].mean()
filtered = len(df) < len(df0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sinh viên", f"{len(df):,}".replace(",", "."), border=True, icon=":material/groups:",
          delta=f"{len(df) / len(df0):.0%} toàn bộ" if filtered else None, delta_color="off",
          delta_arrow="off")
k2.metric("Tỷ lệ bỏ học", C.pct(rate), border=True, icon=":material/trending_down:",
          delta=f"{(rate - base_rate) * 100:+.1f} đ% so với chung" if filtered else None,
          delta_color="inverse")
k3.metric("Tỷ lệ tốt nghiệp", C.pct((df["Target"] == "Graduate").mean()), border=True,
          icon=":material/school:")
k4.metric("Điểm TB HK1 (người có điểm)",
          f"{df.loc[df['Curricular units 1st sem (grade)'] > 0, 'Curricular units 1st sem (grade)'].mean():.2f}",
          border=True, icon=":material/grade:")


def rate_by(frame: pd.DataFrame, col: str, mapper=None, min_n: int = 10) -> pd.DataFrame:
    g = frame.groupby(col)["Dropout"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    g["label"] = g[col].map(mapper) if mapper else g[col].astype(str)
    return g


def rate_bar(g: pd.DataFrame, title: str, horizontal: bool = False, height: int = 280) -> go.Figure:
    colors = [C.INCREASE if m > base_rate else C.SERIES[0] for m in g["mean"]]
    common = dict(marker=dict(color=colors, cornerradius=4), customdata=g["size"],
                  text=[f"{m:.0%}" for m in g["mean"]], textposition="outside", cliponaxis=False)
    if horizontal:
        fig = go.Figure(go.Bar(x=g["mean"] * 100, y=g["label"], orientation="h",
                               hovertemplate="%{y}<br>Tỷ lệ bỏ học %{x:.1f}% · %{customdata} SV<extra></extra>",
                               **common))
        fig.add_vline(x=base_rate * 100, line_dash="dot", line_color=C.NEUTRAL, line_width=1.5)
        fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=GRID,
                         range=[0, max(g["mean"].max() * 100 * 1.18, base_rate * 100 + 5)])
    else:
        fig = go.Figure(go.Bar(x=g["label"], y=g["mean"] * 100,
                               hovertemplate="%{x}<br>Tỷ lệ bỏ học %{y:.1f}% · %{customdata} SV<extra></extra>",
                               **common))
        fig.add_hline(y=base_rate * 100, line_dash="dot", line_color=C.NEUTRAL, line_width=1.5)
        fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor=GRID,
                         range=[0, max(g["mean"].max() * 100 * 1.2, base_rate * 100 + 5)])
    C.style_fig(fig, height=height, title=dict(text=title, font=dict(size=14)),
                bargap=0.45 if not horizontal else 0.3)
    return fig


def card(fig: go.Figure, alt: str) -> None:
    with st.container(border=True):
        C.show_chart(fig, alt=alt)


st.caption(f":material/info: Đường chấm = tỷ lệ bỏ học chung ({base_rate:.0%}). Thanh :red[đỏ] = cao hơn mức chung; "
           ":blue[xanh] = thấp hơn. Nhóm < 10 sinh viên được ẩn để tránh nhiễu.")

VIEWS = {
    "finance": ":material/payments: Tài chính & xã hội",
    "study": ":material/menu_book: Học lực",
    "course": ":material/apartment: Ngành học",
    "profile": ":material/person: Nhân khẩu",
    "corr": ":material/hub: Tương quan",
}
view = st.segmented_control("Chủ đề", list(VIEWS), format_func=VIEWS.get, default="finance",
                            required=True, key="eda_view", label_visibility="collapsed")

yn = {0: "Không", 1: "Có"}
if view == "finance":
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        card(rate_bar(rate_by(df, "Tuition fees up to date", {0: "Chưa đóng đúng hạn", 1: "Đóng đúng hạn"}),
                      "Theo tình trạng học phí"), "Tỷ lệ bỏ học theo tình trạng học phí")
        card(rate_bar(rate_by(df, "Scholarship holder", {0: "Không học bổng", 1: "Có học bổng"}),
                      "Theo học bổng"), "Tỷ lệ bỏ học theo học bổng")
    with c2:
        card(rate_bar(rate_by(df, "Debtor", {0: "Không nợ", 1: "Đang nợ"}), "Theo nợ học phí"),
             "Tỷ lệ bỏ học theo nợ học phí")
        card(rate_bar(rate_by(df, "Displaced", {0: "Ở gần trường", 1: "Học xa nhà"}), "Theo nơi ở"),
             "Tỷ lệ bỏ học theo nơi ở")

elif view == "study":
    c1, c2 = st.columns(2, gap="medium")
    d = df.copy()
    for sem, vi in [("1st", "HK1"), ("2nd", "HK2")]:
        e = d[f"Curricular units {sem} sem (enrolled)"]
        d[f"ratio_{vi}"] = np.where(e > 0, d[f"Curricular units {sem} sem (approved)"] / e.replace(0, 1), np.nan)
    bins = [-0.01, 0.0001, 0.25, 0.5, 0.75, 0.9999, 1.01]
    labels = ["0%", "1–25%", "25–50%", "50–75%", "75–99%", "100%"]
    with c1:
        d["ratio_bin"] = pd.cut(d["ratio_HK1"], bins=bins, labels=labels)
        g = rate_by(d.dropna(subset=["ratio_bin"]), "ratio_bin", None)
        g["label"] = g["ratio_bin"].astype(str)
        card(rate_bar(g, "Theo tỷ lệ qua môn HK1"), "Tỷ lệ bỏ học theo tỷ lệ qua môn học kỳ 1")
    with c2:
        d["ratio_bin2"] = pd.cut(d["ratio_HK2"], bins=bins, labels=labels)
        g = rate_by(d.dropna(subset=["ratio_bin2"]), "ratio_bin2", None)
        g["label"] = g["ratio_bin2"].astype(str)
        card(rate_bar(g, "Theo tỷ lệ qua môn HK2"), "Tỷ lệ bỏ học theo tỷ lệ qua môn học kỳ 2")

    with st.container(border=True):
        sem = st.segmented_control("Học kỳ", ["1st", "2nd"], default="2nd", required=True,
                                   format_func={"1st": "Học kỳ 1", "2nd": "Học kỳ 2"}.get, key="eda_sem")
        col = f"Curricular units {sem} sem (grade)"
        fig = go.Figure()
        for cls in ["Dropout", "Enrolled", "Graduate"]:
            v = df.loc[(df["Target"] == cls) & (df[col] > 0), col]
            fig.add_trace(go.Violin(y=v, name=C.CLASS_VI[cls], line_color=C.CLASS_COLOR[cls],
                                    fillcolor=C.hex_rgba(C.CLASS_COLOR[cls], 0.25), box_visible=True,
                                    meanline_visible=True, points=False, spanmode="hard",
                                    hoverinfo="y+name"))
        n0 = (df[col] == 0).groupby(df["Target"]).mean()
        C.style_fig(fig, height=340, title=dict(text="Phân bố điểm TB (SV có điểm > 0)", font=dict(size=14)))
        fig.update_yaxes(title="Điểm TB (thang 20)", showgrid=True, gridcolor=GRID)
        C.show_chart(fig, alt="Phân bố điểm trung bình theo kết cục")
        st.caption("Tỷ lệ SV có điểm TB = 0 (không đạt môn nào): " + " · ".join(
            f"{C.CLASS_VI[k]} {n0.get(k, 0):.0%}" for k in ["Dropout", "Enrolled", "Graduate"]))

elif view == "course":
    g = rate_by(df, "Course", lambda c: C.label("Course", c)).sort_values("mean")
    card(rate_bar(g, "Tỷ lệ bỏ học theo ngành", horizontal=True, height=40 * len(g) + 80),
         "Tỷ lệ bỏ học theo ngành học")
    with st.container(border=True):
        st.markdown("**Cơ cấu kết cục theo ngành**")
        ct = (pd.crosstab(df["Course"], df["Target"], normalize="index")
              .reindex(columns=["Dropout", "Enrolled", "Graduate"], fill_value=0))
        ct = ct.loc[g["Course"]] if set(g["Course"]).issubset(ct.index) else ct
        fig = go.Figure()
        for cls in ["Dropout", "Enrolled", "Graduate"]:
            fig.add_trace(go.Bar(
                y=[C.label("Course", c) for c in ct.index], x=ct[cls] * 100, orientation="h",
                name=C.CLASS_VI[cls], marker=dict(color=C.CLASS_COLOR[cls],
                                                  line=dict(color="rgba(255,255,255,0.9)", width=2)),
                hovertemplate="%{y}: " + C.CLASS_VI[cls] + " %{x:.1f}%<extra></extra>"))
        C.style_fig(fig, height=34 * len(ct) + 80, legend=True, barmode="stack")
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        C.show_chart(fig, alt="Cơ cấu bỏ học, đang học, tốt nghiệp theo ngành")

elif view == "profile":
    c1, c2 = st.columns(2, gap="medium")
    d = df.copy()
    d["age_bin"] = pd.cut(d["Age at enrollment"], [16, 19, 21, 24, 29, 39, 80],
                          labels=["17–19", "20–21", "22–24", "25–29", "30–39", "40+"])
    with c1:
        g = rate_by(d.dropna(subset=["age_bin"]), "age_bin")
        g["label"] = g["age_bin"].astype(str)
        card(rate_bar(g, "Theo tuổi nhập học"), "Tỷ lệ bỏ học theo nhóm tuổi")
        card(rate_bar(rate_by(df, "Daytime/evening attendance", {1: "Ban ngày", 0: "Buổi tối"}),
                      "Theo ca học"), "Tỷ lệ bỏ học theo ca học")
    with c2:
        card(rate_bar(rate_by(df, "Gender", {0: "Nữ", 1: "Nam"}), "Theo giới tính"),
             "Tỷ lệ bỏ học theo giới tính")
        card(rate_bar(rate_by(df, "Marital status", C.CODEBOOK["Marital status"]),
                      "Theo tình trạng hôn nhân"), "Tỷ lệ bỏ học theo tình trạng hôn nhân")

elif view == "corr":
    num = df[C.FEATURE_COLS + ["Dropout"]]
    corr = num.corr(numeric_only=True)["Dropout"].drop("Dropout").dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=[C.feature_vi(c) for c in top.index], orientation="h",
        marker=dict(color=[C.INCREASE if v > 0 else C.DECREASE for v in top.values], cornerradius=4),
        text=[f"{v:+.2f}" for v in top.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: r = %{x:.3f}<extra></extra>",
    ))
    lim = float(top.abs().max()) * 1.25
    C.style_fig(fig, height=560, title=dict(text="15 đặc trưng tương quan mạnh nhất với việc bỏ học",
                                            font=dict(size=14)))
    fig.update_xaxes(range=[-lim, lim], showgrid=True, gridcolor=GRID, title="Hệ số tương quan Pearson")
    fig.add_vline(x=0, line_width=1, line_color=C.NEUTRAL)
    with st.container(border=True):
        C.show_chart(fig, alt="Hệ số tương quan của các đặc trưng với việc bỏ học")
        st.caption(":red[Dương] = giá trị càng cao càng dễ bỏ học · :blue[âm] = càng cao càng ít bỏ học. "
                   "Tương quan không đồng nghĩa nhân quả.")
