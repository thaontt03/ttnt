import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from common import P

C.page_header(
    "Dự đoán hàng loạt",
    ":material/groups:",
    "Chấm điểm rủi ro cho cả lớp/khoá từ file CSV, lọc danh sách cần can thiệp và xuất báo cáo.",
)
C.require_models()

df_all = C.get_dataset()
model_name = C.active_model()
threshold = C.threshold_for(model_name)


@st.cache_data(show_spinner="Đang đọc file…", max_entries=8)
def parse_csv(raw: bytes) -> pd.DataFrame:
    """Tự nhận dạng dấu phân cách (, ; tab) và mã hoá (UTF-8 / Latin-1)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Không đọc được mã hoá file.")


@st.cache_data(show_spinner="Đang dự đoán…", max_entries=8)
def run_batch(X: pd.DataFrame, name: str, _mt: float) -> pd.DataFrame:
    return P.predict_batch(X, C.get_model(name), C.get_pipeline(), C.get_config(), model_name=name)


# ── Nguồn dữ liệu ───────────────────────────────────────────────────────────
with st.container(border=True):
    top = st.container(horizontal=True, vertical_alignment="bottom")
    with top:
        source = st.segmented_control(
            "Nguồn dữ liệu", ["upload", "sample"], default="upload", required=True,
            format_func={"upload": ":material/upload_file: Tải file CSV",
                         "sample": ":material/dataset: Dữ liệu mẫu"}.get,
            key="batch_source",
        )
        if df_all is not None:
            tmpl = df_all[C.FEATURE_COLS].sample(5, random_state=1).reset_index(drop=True)
            tmpl.insert(0, "Mã SV", [f"SV{i:04d}" for i in range(1, 6)])
            st.download_button("File mẫu CSV", tmpl.to_csv(index=False).encode("utf-8-sig"),
                               file_name="mau_sinh_vien.csv", mime="text/csv",
                               icon=":material/download:", on_click="ignore")
    st.caption(f":material/model_training: Dùng mô hình **{model_name}** · ngưỡng **{threshold:.0%}** "
               "(đổi ở thanh bên trái).")

    raw_df = None
    if source == "upload":
        up = st.file_uploader("Chọn file CSV", type=["csv", "txt"], key="batch_file",
                              help=f"Cần đủ {len(C.FEATURE_COLS)} cột đặc trưng (tên cột như dataset UCI). "
                                   "Dấu phân cách , ; hoặc tab đều được. Các cột khác (mã SV, họ tên, "
                                   "Target…) được giữ nguyên trong kết quả.")
        if up is not None:
            try:
                raw_df = parse_csv(up.getvalue())
            except Exception as e:  # noqa: BLE001
                st.error(f"Không đọc được file: {e}", icon=":material/error:")
    else:
        if df_all is None:
            st.error("Không có dữ liệu mẫu.", icon=":material/error:")
        else:
            with st.container(horizontal=True, vertical_alignment="center"):
                only_test = st.toggle("Chỉ lấy SV thuộc tập test", value=True, key="batch_only_test",
                                      help="Tập test chưa từng dùng để huấn luyện → đối chiếu khách quan.")
                pool = df_all.loc[C.test_index()] if only_test else df_all
                n = st.slider("Số sinh viên lấy ngẫu nhiên (kèm kết cục thật để đối chiếu)",
                              50, len(pool), min(300, len(pool)), step=10, key="batch_n")
            raw_df = (pool.sample(n, random_state=7)[C.FEATURE_COLS + ["Target"]]
                      .rename_axis("Dòng gốc").reset_index())

    with st.expander("Định dạng file yêu cầu", icon=":material/help:"):
        ref = pd.DataFrame({"Tên cột": C.FEATURE_COLS,
                            "Ý nghĩa": [C.feature_vi(c) for c in C.FEATURE_COLS]})
        st.dataframe(ref, hide_index=True, height=260, alt="Danh sách cột bắt buộc")

if raw_df is None:
    st.stop()

# ── Chuẩn hoá & dự đoán ─────────────────────────────────────────────────────
try:
    X, bad_rows = P.prepare_features(raw_df)
except ValueError as e:
    st.error(str(e), title="File thiếu cột", icon=":material/error:")
    st.stop()

if bad_rows:
    st.warning(f"Bỏ qua {len(bad_rows)} dòng có giá trị trống/không phải số "
               f"(dòng: {', '.join(str(i + 2) for i in bad_rows[:15])}{'…' if len(bad_rows) > 15 else ''}).",
               icon=":material/warning:")
X_ok = X.drop(index=bad_rows)
if X_ok.empty:
    st.error("Không còn dòng hợp lệ để dự đoán.", icon=":material/error:")
    st.stop()

pred = run_batch(X_ok, model_name, C._mtime(f"{C.MODELS_DIR}/{P.model_filename(model_name)}"))
extra_cols = [c for c in raw_df.columns if P._norm(c) not in {P._norm(f) for f in C.FEATURE_COLS}]
out = pd.concat([raw_df.loc[X_ok.index, extra_cols], pred], axis=1)

n = len(out)
n_risk = int(out["At_Risk"].sum())
n_watch = int((out["Risk_Level"] == "Cần theo dõi").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sinh viên được chấm", f"{n:,}".replace(",", "."), border=True, icon=":material/groups:")
k2.metric("Có nguy cơ bỏ học", f"{n_risk:,}".replace(",", "."), border=True,
          icon=":material/crisis_alert:", help="P(Bỏ học) ≥ ngưỡng")
k3.metric("Tỷ lệ có nguy cơ", C.pct(n_risk / n), border=True, icon=":material/percent:")
k4.metric("Cần theo dõi thêm", f"{n_watch:,}".replace(",", "."), border=True,
          icon=":material/visibility:", help="Chưa vượt ngưỡng nhưng ≥ 70% ngưỡng")

# ── Biểu đồ ────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("**Phân bố mức rủi ro**")
    lv = out["Risk_Level"].value_counts().reindex(P.RISK_LEVELS, fill_value=0)
    fig = go.Figure(go.Bar(
        x=lv.index, y=lv.values, marker=dict(color=[C.RISK_COLOR[l] for l in lv.index], cornerradius=4),
        text=[f"{v} ({v / n:.0%})" for v in lv.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} SV<extra></extra>",
    ))
    C.style_fig(fig, height=300)
    fig.update_yaxes(title="Số sinh viên", showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                     range=[0, max(lv.values) * 1.2 + 1])
    C.show_chart(fig, alt="Số sinh viên theo mức rủi ro")
with c2, st.container(border=True):
    st.markdown("**Phân bố xác suất bỏ học**")
    fig = go.Figure(go.Histogram(
        x=out["P_Dropout"] * 100, xbins=dict(start=0, end=100, size=5),
        marker=dict(color=C.SERIES[0], line=dict(color="rgba(255,255,255,0.9)", width=1)),
        hovertemplate="%{x}%: %{y} SV<extra></extra>",
    ))
    fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#6250d6", line_width=2,
                  annotation_text=f"Ngưỡng {threshold:.0%}", annotation_position="top right")
    C.style_fig(fig, height=300)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="P(Bỏ học)")
    fig.update_yaxes(title="Số sinh viên", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    C.show_chart(fig, alt="Histogram xác suất bỏ học với vạch ngưỡng")

# ── Đối chiếu nếu file có cột Target ───────────────────────────────────────
tcol = next((c for c in extra_cols if P._norm(c) in ("target", "kết quả thực tế")), None)
if tcol is not None:
    actual = out[tcol].astype(str).str.strip().str.lower().eq("dropout")
    if out[tcol].astype(str).str.strip().str.lower().isin(["dropout", "enrolled", "graduate"]).all():
        predicted = out["At_Risk"].astype(bool)
        tp = int((actual & predicted).sum()); fn = int((actual & ~predicted).sum())
        fp = int((~actual & predicted).sum()); tn = int((~actual & ~predicted).sum())
        with st.container(border=True):
            st.markdown("**:material/fact_check: Đối chiếu với kết cục thực tế (cột Target)**")
            a, b, c, d = st.columns(4)
            a.metric("Bắt được SV bỏ học (Recall)", C.pct(tp / max(tp + fn, 1)), border=True)
            b.metric("Cảnh báo đúng (Precision)", C.pct(tp / max(tp + fp, 1)), border=True)
            c.metric("Bỏ sót", f"{fn} SV", border=True, help="Thực tế bỏ học nhưng không bị cảnh báo")
            d.metric("Báo động nhầm", f"{fp} SV", border=True, help="Bị cảnh báo nhưng thực tế không bỏ học")
            st.caption(f"Ma trận nhầm lẫn: đúng-cảnh báo {tp} · bỏ sót {fn} · báo nhầm {fp} · đúng-an toàn {tn}. "
                       "Nếu dữ liệu có SV thuộc tập train, kết quả sẽ lạc quan hơn thực tế.")

# ── Bảng kết quả ───────────────────────────────────────────────────────────
st.subheader("Danh sách sinh viên", icon=":material/format_list_bulleted:")
with st.container(horizontal=True, vertical_alignment="bottom"):
    levels = st.pills("Lọc mức rủi ro", P.RISK_LEVELS, selection_mode="multi",
                      default=["Cao", "Rất cao"], key="batch_levels")
    courses = sorted(out["Course"].astype(int).unique())
    course_sel = st.multiselect("Ngành", courses, format_func=lambda c: C.label("Course", c),
                                placeholder="Tất cả ngành", key="batch_course", width=360)

view = out
if levels:
    view = view[view["Risk_Level"].isin(levels)]
if course_sel:
    view = view[view["Course"].astype(int).isin(course_sel)]
view = view.sort_values("P_Dropout", ascending=False)

show = pd.DataFrame(index=view.index)
for c in extra_cols:
    show[c] = view[c]
show["P(Bỏ học)"] = view["P_Dropout"]
show["Mức rủi ro"] = view["Risk_Level"]
show["Dự đoán"] = view["Predicted_Class"].map(C.CLASS_VI)
show["Ngành"] = view["Course"].map(lambda c: C.label("Course", c))
show["Điểm HK1"] = view["Curricular units 1st sem (grade)"]
show["Đạt/ĐK HK1"] = (view["Curricular units 1st sem (approved)"].astype(int).astype(str) + "/"
                      + view["Curricular units 1st sem (enrolled)"].astype(int).astype(str))
show["Điểm HK2"] = view["Curricular units 2nd sem (grade)"]
show["Đạt/ĐK HK2"] = (view["Curricular units 2nd sem (approved)"].astype(int).astype(str) + "/"
                      + view["Curricular units 2nd sem (enrolled)"].astype(int).astype(str))
show["Học phí đúng hạn"] = view["Tuition fees up to date"].astype(int).astype(bool)
show["Đang nợ"] = view["Debtor"].astype(int).astype(bool)
show["Học bổng"] = view["Scholarship holder"].astype(int).astype(bool)

st.caption(f"Hiển thị {len(show):,} / {n:,} sinh viên, sắp xếp theo xác suất giảm dần.".replace(",", "."))
st.dataframe(
    show, hide_index=True, height=440, alt="Bảng kết quả dự đoán từng sinh viên",
    column_config={
        "P(Bỏ học)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                                      width="medium"),
        "Điểm HK1": st.column_config.NumberColumn(format="%.1f"),
        "Điểm HK2": st.column_config.NumberColumn(format="%.1f"),
        "Ngành": st.column_config.TextColumn(width="medium"),
    },
)

export = out.copy()
export["Muc_rui_ro"] = export.pop("Risk_Level")
export["Ket_luan"] = export.pop("At_Risk_Label")
with st.container(horizontal=True):
    st.download_button("Tải toàn bộ kết quả", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ket_qua_du_doan.csv", mime="text/csv", type="primary",
                       icon=":material/download:", on_click="ignore")
    st.download_button("Tải danh sách đang lọc", export.loc[view.index].to_csv(index=False).encode("utf-8-sig"),
                       file_name="danh_sach_can_can_thiep.csv", mime="text/csv",
                       icon=":material/filter_alt:", on_click="ignore")
