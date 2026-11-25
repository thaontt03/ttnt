from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from common import P

C.page_header(
    "Dự đoán cá nhân",
    ":material/person_search:",
    "Nhập hồ sơ sinh viên → xác suất bỏ học, mức rủi ro, yếu tố ảnh hưởng và gợi ý can thiệp.",
)
C.require_models()

df = C.get_dataset()
avail = C.available_models()
pipeline = C.get_pipeline()
config = C.get_config()

# ── Khởi tạo form bằng hồ sơ điển hình (1 lần / phiên) ──────────────────────
if C.MACRO_KEY not in st.session_state:
    C.load_student_into_state(C.reference_profile())
st.session_state.setdefault("pred_models", [C.active_model()])
st.session_state.setdefault("pred_result", None)

PRESETS = {
    "typical": "Hồ sơ điển hình",
    "dropout": "SV ngẫu nhiên đã bỏ học",
    "graduate": "SV ngẫu nhiên đã tốt nghiệp",
    "enrolled": "SV ngẫu nhiên đang học",
    "any": "SV ngẫu nhiên bất kỳ",
}


def load_preset(kind: str) -> None:
    if kind == "typical" or df is None:
        C.load_student_into_state(C.reference_profile())
        st.session_state["preset_info"] = None
        return
    pool = df if kind == "any" else df[df["Target"].str.lower() == kind]
    row = pool.sample(1).iloc[0]
    C.load_student_into_state(row)
    st.session_state["preset_info"] = (int(row.name), row["Target"])


# ── Thanh công cụ ───────────────────────────────────────────────────────────
with st.container(horizontal=True, vertical_alignment="bottom"):
    st.pills(
        "Mô hình dự đoán", options=avail, selection_mode="multi", key="pred_models",
        format_func=lambda m: f"{m} ★" if m == C.best_model_name() else m,
        help="Chọn nhiều mô hình để so sánh và lấy kết luận đồng thuận.",
    )
    choice = st.menu_button("Nạp hồ sơ mẫu", options=list(PRESETS), format_func=PRESETS.get,
                            icon=":material/person_add:")
    if choice:
        load_preset(choice)
        st.toast(f"Đã nạp: {PRESETS[choice]}", icon=":material/check:")

info = st.session_state.get("preset_info")
if info:
    st.caption(f":material/badge: Đang xem hồ sơ dòng #{info[0]} trong dữ liệu — kết cục thực tế: "
               f"**{C.CLASS_VI[info[1]]}**")

# Mô hình chính = mô hình đang chọn ở sidebar, sau đó theo thứ hạng CV
_cmp = C.get_comparison()
_rank = {m: i for i, m in enumerate(_cmp["Model"])} if _cmp is not None else {}
selected = sorted((m for m in avail if m in (st.session_state.get("pred_models") or [])),
                  key=lambda m: (m != C.active_model(), _rank.get(m, 99)))

col_form, col_res = st.columns([1.1, 1], gap="large")

# ── FORM ────────────────────────────────────────────────────────────────────
with col_form:
    with st.form("student_form", border=True):
        tabs = st.tabs([":material/menu_book: Học tập", ":material/payments: Tài chính",
                        ":material/person: Cá nhân", ":material/how_to_reg: Tuyển sinh",
                        ":material/family_restroom: Gia đình & bối cảnh"])
        for tab, (tab_name, groups) in zip(tabs, C.FORM_SPEC.items()):
            with tab:
                if tab_name == "Học tập":
                    cols = st.columns(2, gap="medium")
                    for col, (g_name, fields) in zip(cols, groups.items()):
                        with col:
                            st.markdown(f"**{g_name}**")
                            for f, kind, opt in fields:
                                C.render_field(f, kind, opt)
                else:
                    fields = [x for g in groups.values() for x in g]
                    if tab_name == "Tài chính":
                        for f, kind, opt in fields:
                            C.render_field(f, kind, opt)
                        st.caption("Tài chính là nhóm yếu tố có ảnh hưởng mạnh thứ hai sau kết quả học tập.")
                    else:
                        cols = st.columns(2, gap="medium")
                        for i, (f, kind, opt) in enumerate(fields):
                            with cols[i % 2]:
                                C.render_field(f, kind, opt)
                    if tab_name == "Gia đình":
                        st.selectbox("Bối cảnh kinh tế năm nhập học", options=C.macro_options(),
                                     format_func=C.macro_label, key=C.MACRO_KEY,
                                     help="Dữ liệu gốc chỉ có 10 tổ hợp chỉ số vĩ mô (theo năm).")
        submitted = st.form_submit_button(
            "Phân tích nguy cơ", type="primary", icon=":material/analytics:",
            width="stretch", disabled=not selected, shortcut="Ctrl+Enter",
        )
    if not selected:
        st.caption(":material/info: Chọn ít nhất 1 mô hình ở thanh phía trên.")

# ── TÍNH TOÁN ───────────────────────────────────────────────────────────────
if submitted and selected:
    student = C.student_from_state()
    errors, warns = C.validate_student(student)
    if errors:
        st.session_state["pred_result"] = {"errors": errors, "warns": warns}
    else:
        with st.spinner("Đang phân tích…"):
            models = {m: C.get_model(m) for m in selected}
            results = P.predict_multi_model(student, selected, models=models,
                                            pipeline=pipeline, config=config)
            valid = [r for r in results if not r.get("error")]
            primary = valid[0]["model_name"] if valid else None
            expl = (P.explain_by_groups(student, models[primary], pipeline, C.reference_profile())
                    if primary else None)
        pid = datetime.now().strftime("%H%M%S%f")
        st.session_state["pred_result"] = {
            "id": pid, "student": student, "results": results, "primary": primary,
            "consensus": P.consensus(results), "expl": expl, "errors": [], "warns": warns,
            "preset": info,
        }
        if primary:
            r0 = valid[0]
            st.session_state["history"].append({
                "Thời điểm": datetime.now().strftime("%H:%M:%S"),
                "Mô hình": ", ".join(r["model_name"] for r in valid),
                "P(Bỏ học)": r0["dropout_prob"],
                "Mức rủi ro": r0["risk_level"],
                "Kết luận": r0["at_risk_label"],
                "Ngành": C.label("Course", student["Course"]),
                "Điểm HK1": student["Curricular units 1st sem (grade)"],
                "Điểm HK2": student["Curricular units 2nd sem (grade)"],
                "Học phí đúng hạn": "Có" if student["Tuition fees up to date"] else "Không",
                **{f: student[f] for f in C.FEATURE_COLS},
            })

res = st.session_state.get("pred_result")

# ── KẾT QUẢ (cột phải) ─────────────────────────────────────────────────────
with col_res:
    if not res:
        with st.container(border=True, height="stretch", horizontal_alignment="center",
                          vertical_alignment="center"):
            st.space("large")
            st.markdown("## :material/query_stats:", text_alignment="center")
            st.markdown("**Kết quả phân tích sẽ hiển thị ở đây**", text_alignment="center")
            st.caption("Điền hồ sơ (hoặc nạp hồ sơ mẫu) rồi bấm **Phân tích nguy cơ**.",
                       text_alignment="center")
            st.space("large")
    elif res.get("errors"):
        st.error("\n".join(f"- {e}" for e in res["errors"]), title="Dữ liệu chưa hợp lệ",
                 icon=":material/error:")
        for w in res.get("warns", []):
            st.warning(w, icon=":material/warning:")
    else:
        for r in res["results"]:
            if r.get("error"):
                st.warning(f"{r['model_name']}: {r['error']}", icon=":material/warning:")
        valid = [r for r in res["results"] if not r.get("error")]
        if valid:
            r0 = valid[0]
            cons = res["consensus"]
            multi = len(valid) > 1
            at_risk = cons["at_risk"] if multi else r0["at_risk"]

            with st.container(border=True):
                if at_risk:
                    st.markdown("### :red[:material/crisis_alert: Có nguy cơ bỏ học]")
                else:
                    st.markdown("### :blue[:material/verified_user: Không có nguy cơ bỏ học]")
                with st.container(horizontal=True, gap="small"):
                    C.risk_badge(r0["risk_level"])
                    st.badge(f"Ngưỡng {r0['threshold']:.0%}", icon=":material/tune:", color="violet")
                    if multi:
                        st.badge(f"{cons['votes_at_risk']}/{cons['n_models']} mô hình cảnh báo",
                                 icon=":material/how_to_vote:", color="gray")
                st.caption(f"{C.RISK_DESC[r0['risk_level']]}  ·  Mô hình chính: **{r0['model_name']}**")
                C.show_chart(C.gauge_figure(r0["dropout_prob"], r0["threshold"]),
                             alt=f"Đồng hồ xác suất bỏ học {r0['dropout_prob']:.0%}",
                             key=f"gauge_{res['id']}")
                st.markdown("**Xác suất theo kết cục**")
                C.show_chart(C.proba_bar(r0["probabilities"]),
                             alt="Thanh xác suất bỏ học, đang học, tốt nghiệp",
                             key=f"pbar_{res['id']}")
                if res.get("preset"):
                    actual = res["preset"][1]
                    hit = (actual == "Dropout") == at_risk
                    st.caption(f"{':material/check_circle:' if hit else ':material/cancel:'} "
                               f"Kết cục thực tế của hồ sơ mẫu: **{C.CLASS_VI[actual]}** — "
                               f"dự đoán {'khớp' if hit else 'chưa khớp'}.")
            for w in res.get("warns", []):
                st.warning(w, icon=":material/warning:")

# ── PHÂN TÍCH CHI TIẾT (toàn chiều rộng) ─────────────────────────────────────
if res and not res.get("errors") and res.get("primary"):
    valid = [r for r in res["results"] if not r.get("error")]
    student, primary = res["student"], res["primary"]
    st.space("small")

    views = {
        "factors": ":material/bar_chart: Yếu tố ảnh hưởng",
        "whatif": ":material/science: Mô phỏng what-if",
        "advice": ":material/tips_and_updates: Khuyến nghị",
    }
    if len(valid) > 1:
        views = {"compare": ":material/compare: So sánh mô hình", **views}
    view = st.segmented_control("Phân tích chi tiết", options=list(views), format_func=views.get,
                                default=list(views)[0], key="pred_view", required=True,
                                label_visibility="collapsed")

    if view == "compare":
        cons = res["consensus"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Đồng thuận", f"{cons['agreement']:.0%}", border=True,
                  help="Tỷ lệ mô hình cùng kết luận với kết luận chung")
        c2.metric("P(Bỏ học) trung bình", C.pct(cons["mean_prob"]), border=True)
        c3.metric("Biên trung bình so với ngưỡng", f"{cons['mean_margin'] * 100:+.1f} đ%", border=True,
                  help="Trung bình (P − ngưỡng) của các mô hình; dương = vượt ngưỡng")
        if cons["tie_broken"]:
            st.caption(":material/balance: Số phiếu hoà — kết luận chung dựa trên biên trung bình.")

        d = pd.DataFrame(valid).sort_values("dropout_prob")
        fig = go.Figure()
        for _, r in d.iterrows():
            fig.add_shape(type="line", x0=r["threshold"] * 100, x1=r["dropout_prob"] * 100,
                          y0=r["model_name"], y1=r["model_name"],
                          line=dict(color=C.NEUTRAL, width=2))
        fig.add_trace(go.Scatter(
            x=d["threshold"] * 100, y=d["model_name"], mode="markers", name="Ngưỡng của mô hình",
            marker=dict(symbol="line-ns", size=18, line=dict(width=3, color="#6250d6")),
            hovertemplate="%{y}: ngưỡng %{x:.0f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=d["dropout_prob"] * 100, y=d["model_name"], mode="markers+text", name="P(Bỏ học)",
            marker=dict(size=14, color=[C.RISK_COLOR[l] for l in d["risk_level"]],
                        line=dict(width=2, color="white")),
            text=[f"{p * 100:.0f}%" for p in d["dropout_prob"]], textposition="middle right",
            customdata=d["risk_level"],
            hovertemplate="%{y}: P = %{x:.1f}% · %{customdata}<extra></extra>",
        ))
        C.style_fig(fig, height=60 * len(d) + 70, legend=True)
        fig.update_xaxes(range=[0, 105], ticksuffix="%", showgrid=True,
                         gridcolor="rgba(128,128,128,0.15)", title="Xác suất bỏ học")
        C.show_chart(fig, alt="So sánh xác suất bỏ học và ngưỡng của từng mô hình",
                     key=f"cmp_{res['id']}")

        table = pd.DataFrame([{
            "Mô hình": r["model_name"], "P(Bỏ học)": r["dropout_prob"], "Ngưỡng": r["threshold"],
            "Mức rủi ro": r["risk_level"], "Kết luận": r["at_risk_label"],
            "Đang học": r["probabilities"]["Enrolled"], "Tốt nghiệp": r["probabilities"]["Graduate"],
        } for r in valid])
        st.dataframe(table, hide_index=True, alt="Bảng kết quả theo từng mô hình", column_config={
            "P(Bỏ học)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Ngưỡng": st.column_config.NumberColumn(format="percent"),
            "Đang học": st.column_config.NumberColumn(format="percent"),
            "Tốt nghiệp": st.column_config.NumberColumn(format="percent"),
        })

    elif view == "factors":
        expl = res["expl"]
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown(f"**Đóng góp của từng nhóm yếu tố** · {primary}")
                if expl["impact"].abs().max() < 0.005:
                    st.info("Hồ sơ này gần như trùng với hồ sơ điển hình nên không nhóm yếu tố nào "
                            "tạo khác biệt đáng kể. Hãy thay đổi vài thông tin hoặc nạp một sinh viên "
                            "ngẫu nhiên để thấy rõ tác động.", icon=":material/info:")
                C.show_chart(C.impact_figure(expl, top=10),
                             alt="Biểu đồ mức đóng góp của các nhóm yếu tố vào xác suất bỏ học",
                             key=f"imp_{res['id']}")
                p_ref, p_now = expl.attrs.get("p_ref"), expl.attrs.get("p")
                if p_ref is not None:
                    st.caption(f"Hồ sơ điển hình (trung vị/mode toàn bộ dữ liệu) có P = **{C.pct(p_ref)}**; "
                               f"sinh viên này **{C.pct(p_now)}**. Giá trị Shapley chia phần chênh lệch "
                               f"**{(p_now - p_ref) * 100:+.1f} đ%** cho các nhóm yếu tố — tổng các thanh đúng "
                               "bằng chênh lệch đó. :red[Đỏ] = làm tăng rủi ro, :blue[xanh] = giúp giảm.")
        with right:
            up = expl[expl["impact"] > 0.005].head(3)
            down = expl[expl["impact"] < -0.005].head(3)
            with st.container(border=True, height="stretch"):
                st.markdown("**:red[:material/arrow_upward:] Đẩy rủi ro lên**")
                if len(up):
                    for g in up.itertuples():
                        st.markdown(f"- {g.group} :red[**{g.impact * 100:+.1f} đ%**]")
                else:
                    st.caption("Không có nhóm nào đáng kể.")
                st.markdown("**:blue[:material/arrow_downward:] Kéo rủi ro xuống**")
                if len(down):
                    for g in down.itertuples():
                        st.markdown(f"- {g.group} :blue[**{g.impact * 100:+.1f} đ%**]")
                else:
                    st.caption("Không có nhóm nào đáng kể.")
                with st.expander("Hồ sơ đã nhập", icon=":material/description:"):
                    prof = pd.DataFrame({
                        "Thuộc tính": [C.feature_vi(f) for f in C.FEATURE_COLS],
                        "Giá trị": [C.label(f, student[f]) if f in C.CODEBOOK
                                    else f"{student[f]:g}" for f in C.FEATURE_COLS],
                    })
                    st.dataframe(prof, hide_index=True, height=300, alt="Hồ sơ sinh viên đã nhập")

    elif view == "whatif":
        @st.fragment
        def what_if(base: dict, model_name: str, pid: str) -> None:
            model = C.get_model(model_name)
            t = C.threshold_for(model_name)
            p0 = P.predict_single(base, model, pipeline, config, model_name)["dropout_prob"]
            st.markdown(f"Thay đổi các yếu tố *có thể can thiệp* để xem rủi ro thay đổi ra sao "
                        f"(mô hình **{model_name}**). Chỉ phần này được tính lại.")
            s = dict(base)
            c1, c2, c3 = st.columns([1, 1, 1.2], gap="medium")
            with c1, st.container(border=True, height="stretch"):
                st.markdown("**Tài chính**")
                s["Tuition fees up to date"] = float(st.toggle(
                    "Đóng học phí đúng hạn", value=bool(base["Tuition fees up to date"]), key=f"wi_t_{pid}"))
                s["Debtor"] = float(st.toggle("Đang nợ học phí", value=bool(base["Debtor"]), key=f"wi_d_{pid}"))
                s["Scholarship holder"] = float(st.toggle(
                    "Có học bổng", value=bool(base["Scholarship holder"]), key=f"wi_s_{pid}"))
            with c2, st.container(border=True, height="stretch"):
                st.markdown("**Học kỳ 2**")
                e2 = int(base["Curricular units 2nd sem (enrolled)"])
                if e2 > 0:
                    a2 = st.slider("Số môn đạt", 0, e2, int(base["Curricular units 2nd sem (approved)"]),
                                   key=f"wi_a2_{pid}")
                    g_def = float(base["Curricular units 2nd sem (grade)"]) or 12.0
                    g2 = st.slider("Điểm trung bình", 9.5, 20.0, float(np.clip(g_def, 9.5, 20)), 0.1,
                                   key=f"wi_g2_{pid}", disabled=a2 == 0,
                                   help="Điểm TB chỉ có nghĩa khi có môn đạt (thang 20, đạt ≈ 10)")
                    s["Curricular units 2nd sem (approved)"] = float(a2)
                    s["Curricular units 2nd sem (grade)"] = float(g2) if a2 > 0 else 0.0
                    s["Curricular units 2nd sem (evaluations)"] = float(
                        max(base["Curricular units 2nd sem (evaluations)"], a2))
                else:
                    st.caption("Sinh viên không đăng ký môn nào ở HK2.")
            p1 = P.predict_single(s, model, pipeline, config, model_name)
            with c3, st.container(border=True, height="stretch"):
                st.markdown("**Kết quả mô phỏng**")
                diff = round((p1["dropout_prob"] - p0) * 100, 1)
                st.metric("P(Bỏ học) sau thay đổi", C.pct(p1["dropout_prob"]),
                          delta=diff if diff != 0 else 0, delta_color="inverse",
                          delta_description="điểm % so với ban đầu")
                C.risk_badge(p1["risk_level"])
                st.caption(f"Ban đầu {C.pct(p0)} · ngưỡng {t:.0%}")
                if p1["at_risk"] and p0 < t:
                    st.error("Vượt ngưỡng cảnh báo!", icon=":material/swap_horiz:")
                elif not p1["at_risk"] and p0 >= t:
                    st.success("Đã xuống dưới ngưỡng cảnh báo!", icon=":material/swap_horiz:")

        what_if(student, primary, res["id"])

    elif view == "advice":
        r0 = valid[0]
        recs = C.recommendations(student, r0)
        cols = st.columns(min(3, len(recs)), gap="medium")
        for i, (icon, title, text) in enumerate(recs):
            with cols[i % len(cols)], st.container(border=True, height="stretch"):
                st.markdown(f"**{icon} {title}**")
                st.markdown(text)
        st.caption(":material/info: Gợi ý mang tính tham khảo, dựa trên quy tắc từ phân tích dữ liệu — "
                   "không thay thế đánh giá của cố vấn học tập.")

# ── LỊCH SỬ ────────────────────────────────────────────────────────────────
hist = st.session_state["history"]
if hist:
    with st.expander(f"Lịch sử dự đoán trong phiên ({len(hist)})", icon=":material/history:"):
        h = pd.DataFrame(hist)
        st.dataframe(
            h[["Thời điểm", "Mô hình", "P(Bỏ học)", "Mức rủi ro", "Kết luận", "Ngành",
               "Điểm HK1", "Điểm HK2", "Học phí đúng hạn"]].iloc[::-1],
            hide_index=True, alt="Lịch sử các lượt dự đoán",
            column_config={"P(Bỏ học)": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1)},
        )
        with st.container(horizontal=True):
            st.download_button("Tải lịch sử (CSV)", h.to_csv(index=False).encode("utf-8-sig"),
                               file_name="lich_su_du_doan.csv", mime="text/csv",
                               icon=":material/download:", on_click="ignore")
            if st.button("Xoá lịch sử", icon=":material/delete:", type="tertiary"):
                st.session_state["history"] = []
                st.rerun()
