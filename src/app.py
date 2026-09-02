# -*- coding: utf-8 -*-
"""HyFinEval 最简 Web UI（Streamlit）。

运行：
    pip install streamlit
    streamlit run app.py
设置环境变量 HY3_API_KEY 后，勾选「使用 Hy3 真实模型」即可调用腾讯混元 Hy3；
未设置 key 时取消勾选，自动用基线示例（基于本地真实指标，无需联网）。

UI 复用 hy3_app.generate() 产出结构化输出（answer + citations），
并实时调用 evaluator 给出 7 维评分，直观展示「应用 → 引用溯源 → 评分」闭环。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import config
import data_store
import hy3_app
import baseline_app
import evaluator


def _parse_pdf(uploaded):
    """解析上传的 PDF 为 [{page, text}]；缺依赖或失败时返回 None/[]。"""
    try:
        import pdfplumber
    except ImportError:
        st.error("未安装 pdfplumber，无法解析 PDF。请先 `pip install pdfplumber`。")
        return None
    try:
        pages = []
        with pdfplumber.open(uploaded) as pdf:
            for i, pg in enumerate(pdf.pages, 1):
                txt = pg.extract_text() or ""
                if txt.strip():
                    pages.append({"page": i, "text": txt})
        return pages
    except Exception as e:
        st.error(f"PDF 解析失败：{e}")
        return None


st.set_page_config(page_title="HyFinEval · 财报速读助手", layout="wide")
st.title("📊 HyFinEval · 财报速读助手")
st.caption(
    "基于腾讯混元 Hy3 的金融分析应用：指标提取 / 财务问答 / 公告摘要，输出带引用、可溯源；"
    "并实时给出 7 维评分。面向基金 / 行业分析师的「财报速读」真实用户场景。"
)

# 从指标库动态生成「代码 / 年份」可选项，避免误选库外组合（导致闭卷、评分失真）
_IND = data_store.get_indicators()
_CODE_YEARS = {}
for _k in _IND:
    _c, _y = _k.split("_", 1)
    _CODE_YEARS.setdefault(_c, []).append(_y)
_CODES = sorted(_CODE_YEARS.keys())
def _code_label(c):
    nm = data_store.company_name(c)
    return f"{nm}（{c}）" if nm != c else c

col_left, col_right = st.columns([1, 2])

with col_left:
    default_code = "600519" if "600519" in _CODES else (_CODES[0] if _CODES else "")
    code = st.selectbox("股票代码", _CODES, index=_CODES.index(default_code),
                        format_func=_code_label,
                        help="仅列出指标库中真实存在的股票；年份随所选股票联动。")
    _years = sorted(_CODE_YEARS.get(code, []), reverse=True)
    year = st.selectbox("报告期（年）", _years, index=0,
                        help="仅列出该股票在指标库中已有的报告期。")
    subtask = st.selectbox("任务类型", ["财报指标提取", "财务问答", "公告摘要"])
    use_rag = st.checkbox(
        "开卷（RAG 注入真实指标表）",
        value=True,
        help="勾选：注入真实指标表，回答可溯源、数值可验证；取消：闭卷模式，模型凭自身知识作答（具体数值未经指标库核实）。",
    )
    use_hy3 = st.checkbox("使用 Hy3 真实模型", value=config.USE_HY3,
                          help="取消勾选则使用基线示例（无需 API key）")
    if not config.USE_HY3:
        st.warning("⚠️ 未检测到 HY3_API_KEY 环境变量，将以基线示例运行（输出基于本地真实指标，无需联网）。"
                   "设置 key 后即可勾选调用真实 Hy3。")
    if use_rag:
        st.info(f"✅ 已加载 {data_store.company_name(code)}（{code}）{year}年真实指标，开卷 RAG 模式。")
    else:
        st.info("ℹ️ 闭卷模式：未注入指标表，模型凭自身知识作答，具体数值未经指标库核实。")
    custom_q = st.text_area(
        "自定义问题",
        value="",
        height=80,
        help="留空则用上方『代码+年份+任务』生成的模板；若填写，会自动附带已选的代码与年份，无需重复输入。",
    )

    # ---- 上传财报/研报 PDF（可选，覆盖任意股票/年份，按页码溯源）----
    uploaded = st.file_uploader(
        "上传财报 / 研报 PDF（可选，覆盖任意公司）",
        type=["pdf"],
        help="上传后将以文档内容为 RAG 上下文，回答按页码引用。上传文档与上方指标库可叠加使用。",
    )
    pdf_pages = None
    pdf_name = None
    if uploaded is not None:
        _key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("pdf_key") != _key:
            with st.spinner("解析 PDF 中…"):
                pdf_pages = _parse_pdf(uploaded)
            st.session_state["pdf_pages"] = pdf_pages
            st.session_state["pdf_key"] = _key
            st.session_state["pdf_name"] = uploaded.name
        pdf_pages = st.session_state.get("pdf_pages")
        pdf_name = st.session_state.get("pdf_name")
        if pdf_pages:
            st.info(f"✅ 已解析上传文档：{pdf_name}（{len(pdf_pages)} 页），将作为 RAG 上下文按页码引用。")
        elif pdf_pages == []:
            st.warning("该 PDF 未提取到文本（可能是扫描件/图片型），无法用于 RAG；可改用指标库或闭卷模式。")

    go = st.button("生成", type="primary")

prefix = f"{data_store.company_name(code)}（{code}）{year}年"
if subtask == "财报指标提取":
    tmpl = f"请从{prefix}财务数据中提取【扣非净利润】的数值。"
elif subtask == "财务问答":
    tmpl = f"请基于{prefix}财务数据，分析其当年盈利能力与偿债压力。"
else:
    tmpl = f"请摘要{prefix}相关公告的关键要素，不编造具体数字。"
custom = custom_q.strip()
if custom:
    # 自定义问题自动带上上方已选的代码与年份，避免 RAG 丢失定位信息
    question = f"请基于{prefix}财务数据回答：{custom}"
else:
    question = tmpl

if go:
    sample = {"subtask": subtask, "input": question, "difficulty": "中"}
    with st.spinner("Hy3 思考中…" if use_hy3 else "基线生成中…"):
        if use_hy3:
            out = hy3_app.generate(sample, use_rag=use_rag, pdf_pages=pdf_pages, pdf_name=pdf_name)
            if not out:
                st.warning("Hy3 调用未返回结果（可能未配置 key 或网络异常），已自动回退到基线示例。")
                out = baseline_app.baseline_generate(sample)
        else:
            out = baseline_app.baseline_generate(sample)

    with col_right:
        st.subheader("① 输入")
        st.write(question)

        if not out:
            st.error(
                "未获取到输出。使用 Hy3 时请确认：① 已设置环境变量 HY3_API_KEY；"
                "② HY3_BASE_URL 指向可用 endpoint（如 tokenhub-intl.tencentmaas.com/v1）。"
            )
        else:
            st.subheader("② 模型输出")
            st.markdown(out.get("answer", ""))

            cites = out.get("citations") or []
            if cites:
                st.subheader("③ 引用 / 溯源")
                rows = []
                for c in cites:
                    if not isinstance(c, dict):
                        continue
                    if c.get("page") or c.get("source") in ("pdf", "report"):
                        rows.append({
                            "来源": f"文档·第{c.get('page', '?')}页",
                            "字段/摘录": c.get("excerpt") or c.get("field") or "",
                            "数值": c.get("value", ""),
                        })
                    else:
                        rows.append({
                            "来源": f"{data_store.company_name(c.get('company', ''))} {c.get('year', '')}",
                            "字段/摘录": c.get("field") or "",
                            "数值": c.get("value", ""),
                        })
                st.table(rows)
            else:
                st.info("本次输出未携带结构化 citations（可能为闭卷或基线模式）。")

            # 实时评估：复用同一套 7 维 rubric，离线可跑
            res = evaluator.evaluate(sample, out)
            dims = res["dimensions"]
            st.subheader(f"④ 7 维评分（综合 {res['overall']}）")
            for d, w in config.DIMENSION_WEIGHTS.items():
                v = dims.get(d, 0)
                st.progress(v, text=f"{config.DIMENSION_NAMES[d]}（权重 {w}）：{v:.2f}")
            st.caption("失败模式：" + res["failure_mode"])

st.markdown("---")
st.caption("HyFinEval · 犀牛鸟开源实战任务个人作品 · 模型：腾讯混元 Hy3 / HunYuan")
