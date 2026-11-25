import plotly.graph_objects as go
import streamlit as st

import common as C

C.page_header(
    "Hệ thống cảnh báo sớm nguy cơ bỏ học",
    ":material/school:",
    "Ứng dụng học máy giúp cố vấn học tập phát hiện sớm sinh viên có nguy cơ bỏ học "
    "dựa trên kết quả học tập, tài chính và hoàn cảnh cá nhân.",
)

df = C.get_dataset()
cmp_df = C.get_comparison()
best = C.best_model_name()

# ── KPI ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
if df is not None:
    k1.metric("Sinh viên trong dữ liệu", f"{len(df):,}".replace(",", "."), border=True,
              icon=":material/groups:", height="stretch",
              help="Dữ liệu sinh viên đại học Bồ Đào Nha (UCI ML Repository)")
    k2.metric("Tỷ lệ bỏ học thực tế", C.pct(df["Dropout"].mean()), border=True,
              icon=":material/trending_down:", height="stretch")
else:
    k1.metric("Sinh viên trong dữ liệu", "—", border=True, height="stretch")
    k2.metric("Tỷ lệ bỏ học thực tế", "—", border=True, height="stretch")

row = None
if cmp_df is not None and best in set(cmp_df["Model"]) and "test_f1_macro" in cmp_df:
    row = cmp_df.set_index("Model").loc[best]
    k3.metric("Phát hiện đúng SV bỏ học", C.pct(row["test_recall_dropout"]), border=True,
              icon=":material/radar:", height="stretch",
              help=f"Recall(Dropout) của {best} trên tập test độc lập. Mục tiêu ≥ 75%.")
    k4.metric("F1-macro (tập test)", f"{row['test_f1_macro']:.3f}", border=True,
              icon=":material/verified:", height="stretch",
              help=f"Mô hình {best}, ngưỡng {row['threshold']:.0%}. Mục tiêu ≥ 0.70.")
else:
    k3.metric("Phát hiện đúng SV bỏ học", "—", border=True, height="stretch")
    k4.metric("F1-macro (tập test)", "—", border=True, height="stretch")

st.space("small")

# ── Lối tắt ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
cards = [
    (c1, "app_pages/predict.py", ":material/person_search:", "Dự đoán cá nhân",
     "Nhập hồ sơ 1 sinh viên, so sánh nhiều mô hình, xem yếu tố ảnh hưởng và mô phỏng *what-if*."),
    (c2, "app_pages/batch.py", ":material/groups:", "Dự đoán hàng loạt",
     "Tải lên CSV cả lớp/khoá, lọc danh sách cần can thiệp và xuất báo cáo."),
    (c3, "app_pages/eda.py", ":material/insights:", "Khám phá dữ liệu",
     "Tỷ lệ bỏ học theo ngành, tài chính, học lực… với bộ lọc tương tác."),
]
for col, path, icon, title, desc in cards:
    with col.container(border=True, height="stretch"):
        st.subheader(title, icon=icon)
        st.markdown(desc)
        st.space("stretch")
        st.page_link(path, label="Mở trang", icon=":material/arrow_forward:")

# ── Cách hệ thống hoạt động + phân bố kết quả ────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")
with left:
    with st.container(border=True, height="stretch"):
        st.subheader("Cách hệ thống ra quyết định", icon=":material/account_tree:")
        t = C.threshold_for(best)
        st.markdown(
            f"""
1. **Mô hình** ({best or '—'}) ước lượng xác suất sinh viên thuộc 3 nhóm:
   :red-badge[Bỏ học] :orange-badge[Đang học] :blue-badge[Tốt nghiệp].
2. **Ngưỡng cảnh báo** riêng cho từng mô hình (của {best or '—'}: **{t:.0%}**) được chọn trên
   dự đoán *out-of-fold* của tập train sao cho bắt được ≥ 78% SV bỏ học, rồi mới kiểm chứng
   trên tập test độc lập.
3. **Kết luận**: *Có nguy cơ* ⇔ P(Bỏ học) ≥ ngưỡng. Mức rủi ro tính tương đối theo ngưỡng:
   {C.risk_md('Thấp')} {C.risk_md('Cần theo dõi')} {C.risk_md('Cao')} {C.risk_md('Rất cao')}
4. **Giải thích**: giá trị *Shapley* theo 13 nhóm yếu tố cho biết mỗi nhóm đẩy rủi ro lên/xuống bao nhiêu so với một sinh viên điển hình.
"""
        )
with right:
    with st.container(border=True, height="stretch"):
        st.subheader("Kết cục học tập trong dữ liệu", icon=":material/donut_large:")
        if df is not None:
            counts = df["Target"].value_counts().reindex(["Dropout", "Enrolled", "Graduate"])
            fig = go.Figure(go.Pie(
                labels=[C.CLASS_VI[c] for c in counts.index], values=counts.values,
                hole=0.62, sort=False, direction="clockwise",
                marker=dict(colors=[C.CLASS_COLOR[c] for c in counts.index],
                            line=dict(color="rgba(255,255,255,0.9)", width=2)),
                textinfo="percent", textfont=dict(size=13, color="white"),
                hovertemplate="%{label}: %{value} SV (%{percent})<extra></extra>",
            ))
            C.style_fig(fig, height=260, legend=True)
            fig.update_layout(legend=dict(orientation="v", x=1, y=0.5, yanchor="middle"),
                              annotations=[dict(text=f"<b>{len(df):,}</b><br>sinh viên".replace(",", "."),
                                                showarrow=False, font=dict(size=15))])
            C.show_chart(fig, alt="Biểu đồ tròn tỷ lệ bỏ học, đang học, tốt nghiệp")
        else:
            st.info("Chưa có file data/raw/dataset.csv", icon=":material/info:")

if row is not None:
    st.caption(f":material/info: Chỉ số đo trên {C.get_config().get('n_test', 885)} sinh viên "
               "của tập test chưa từng dùng để huấn luyện hay chọn ngưỡng.")
