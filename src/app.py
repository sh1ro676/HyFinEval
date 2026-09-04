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


# --------------------------------------------------------------------------- #
# 主题 / 样式
# --------------------------------------------------------------------------- #
_CSS = """
<style>
  .block-container {padding-top: 1.2rem;}
  .hy-banner {
    background: linear-gradient(135deg, #0f4c81 0%, #1b6aa5 55%, #2a9d8f 100%);
    padding: 26px 30px; border-radius: 16px; color: #ffffff;
    margin-bottom: 6px; box-shadow: 0 6px 18px rgba(15, 76, 129, .20);
  }
  .hy-banner h1 {margin: 0; font-size: 1.75rem; letter-spacing: .5px; font-weight: 800;}
  .hy-banner p {margin: 8px 0 0 0; opacity: .93; font-size: .94rem; line-height: 1.55;}

  .hy-pills {margin: 8px 0 4px 0;}
  .hy-pill {display: inline-block; padding: 3px 13px; border-radius: 999px;
            font-size: .78rem; margin: 0 8px 6px 0; font-weight: 600; white-space: nowrap;}
  .hy-pill.rag   {background:#e6f4ea; color:#1e7d34; border:1px solid #bfe3c6;}
  .hy-pill.closed{background:#fdecea; color:#c0392b; border:1px solid #f5c6c0;}
  .hy-pill.model {background:#eef3ff; color:#2b5cc4; border:1px solid #cdd9f7;}
  .hy-pill.base  {background:#fff4e0; color:#9a6b00; border:1px solid #f2d9a4;}

  .hy-card {background:#ffffff; border:1px solid #e6e9ef; border-radius:14px;
            padding:18px 20px; box-shadow:0 1px 4px rgba(0,0,0,.05); margin-bottom:16px;}
  .hy-card h4 {margin:0 0 10px 0; color:#1f2733; font-size:1.02rem; font-weight:700;}

  .hy-score-wrap {display:flex; align-items:center; gap:24px; flex-wrap:wrap;}
  .hy-score {font-size:3.3rem; font-weight:800; line-height:1;}
  .hy-score .u {font-size:1.15rem; color:#8a93a5; font-weight:600;}
  .hy-score-name {font-weight:700; font-size:1.05rem; color:#1f2733;}
  .hy-fm {margin-top:4px; font-size:.86rem; color:#6b7280;}

  .hy-dim {margin:11px 0;}
  .hy-dim .lab {display:flex; justify-content:space-between; font-size:.83rem;
                color:#3c4454; margin-bottom:4px;}
  .hy-dim .lab .wt {color:#9aa2b1; font-variant-numeric:tabular-nums;}
  .hy-bar {background:#eef1f6; border-radius:8px; height:12px; overflow:hidden;}
  .hy-bar > span {display:block; height:100%; border-radius:8px;
                  transition:width .4s ease;}

  .hy-empty {border:1.5px dashed #cdd4e0; border-radius:14px; padding:44px 30px;
             text-align:center; color:#7b869a; background:#fbfcfe;}
  .hy-empty .big {font-size:2.2rem; margin-bottom:10px;}

  .hy-cit {background:#ffffff; border:1px solid #e6e9ef; border-left:4px solid #2a9d8f;
           border-radius:10px; padding:12px 16px; margin-bottom:10px;
           box-shadow:0 1px 3px rgba(0,0,0,.04);}
  .hy-cit .src {font-size:.78rem; color:#5b6574; font-weight:600;}
  .hy-cit .field {font-size:.93rem; color:#1f2733; margin:3px 0;}
  .hy-cit .val {font-size:.97rem; font-weight:700; color:#0f4c81;}

  .hy-foot {margin-top:8px; text-align:center; color:#9aa2b1; font-size:.8rem;}
</style>
"""


def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def _score_color(v01):
    """0..1 分数 -> 颜色（绿=优 / 黄=中 / 红=差）。"""
    if v01 >= 0.85:
        return "#1e7d34"
    if v01 >= 0.7:
        return "#d9a300"
    return "#c0392b"


def _banner():
    st.markdown(
        '<div class="hy-banner">'
        "<h1>📊 HyFinEval · 财报速读助手</h1>"
        "<p>基于腾讯混元 Hy3 的金融分析应用：指标提取 / 财务问答 / 公告摘要，"
        "输出带引用、可溯源，并实时给出 7 维评分。面向基金 / 行业分析师的「财报速读」场景。</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _mode_pills(use_rag, use_hy3):
    rag = ('<span class="hy-pill rag">开卷 · RAG 注入真实指标</span>'
           if use_rag else '<span class="hy-pill closed">闭卷 · 凭模型自身知识</span>')
    mdl = ('<span class="hy-pill model">混元 Hy3 真实模型</span>'
           if use_hy3 else '<span class="hy-pill base">基线示例（无需 key）</span>')
    st.markdown(f'<div class="hy-pills">{rag}{mdl}</div>', unsafe_allow_html=True)


def _empty_state():
    st.markdown(
        '<div class="hy-empty">'
        '<div class="big">🧭</div>'
        '<div style="font-weight:600;font-size:1.02rem;color:#5b6574;margin-bottom:6px;">'
        "在左侧选好「股票 / 年份 / 任务」，点击「生成」开始</div>"
        "<div style=\"font-size:.86rem;\">可切换开卷/闭卷、上传研报 PDF，生成后这里展示 "
        "模型输出 · 引用溯源 · 7 维评分。</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 页面骨架
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="HyFinEval · 财报速读助手", layout="wide")
_inject_css()
_banner()

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


# ---- 左侧：配置（侧边栏）----
with st.sidebar:
    st.header("⚙️ 查询配置")
    default_code = "600519" if "600519" in _CODES else (_CODES[0] if _CODES else "")
    code = st.selectbox("股票代码", _CODES, index=_CODES.index(default_code),
                        format_func=_code_label,
                        help="仅列出指标库中真实存在的股票；年份随所选股票联动。")
    _years = sorted(_CODE_YEARS.get(code, []), reverse=True)
    year = st.selectbox("报告期（年）", _years, index=0,
                        help="仅列出该股票在指标库中已有的报告期。")
    subtask = st.selectbox("任务类型", ["财报指标提取", "财务问答", "公告摘要"])

    st.divider()
    st.subheader("模式")
    use_rag = st.checkbox(
        "开卷（RAG 注入真实指标表）", value=True,
        help="勾选：注入真实指标表，回答可溯源、数值可验证；取消：闭卷模式，模型凭自身知识作答（具体数值未经指标库核实）。",
    )
    use_hy3 = st.checkbox("使用 Hy3 真实模型", value=config.USE_HY3,
                          help="取消勾选则使用基线示例（无需 API key）")
    if not config.USE_HY3:
        st.warning("⚠️ 未检测到 HY3_API_KEY，将以基线示例运行。设置 key 后可调用真实 Hy3。")

    st.divider()
    custom_q = st.text_area(
        "自定义问题", value="", height=84,
        help="留空则用上方『代码+年份+任务』生成的模板；若填写，会自动附带已选代码与年份。",
    )

    uploaded = st.file_uploader(
        "上传财报 / 研报 PDF（可选）", type=["pdf"],
        help="上传后以文档内容为 RAG 上下文，按页码引用；与指标库可叠加。",
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
            st.caption(f"✅ 已解析：{pdf_name}（{len(pdf_pages)} 页）")
        elif pdf_pages == []:
            st.warning("该 PDF 未提取到文本（可能是扫描件/图片型）。")

    st.divider()
    go = st.button("🚀 生成", type="primary")

# ---- 构造问题 ----
prefix = f"{data_store.company_name(code)}（{code}）{year}年"
if subtask == "财报指标提取":
    tmpl = f"请从{prefix}财务数据中提取【扣非净利润】的数值。"
elif subtask == "财务问答":
    tmpl = f"请基于{prefix}财务数据，分析其当年盈利能力与偿债压力。"
else:
    tmpl = f"请摘要{prefix}相关公告的关键要素，不编造具体数字。"
custom = custom_q.strip()
question = f"请基于{prefix}财务数据回答：{custom}" if custom else tmpl

# ---- 右侧：状态 + 结果 ----
_mode_pills(use_rag, use_hy3)
if use_rag:
    st.caption(f"✅ 已加载 {data_store.company_name(code)}（{code}）{year} 年真实指标（开卷）。")
else:
    st.caption("ℹ️ 闭卷模式：未注入指标表，具体数值未经指标库核实。")

if not go:
    _empty_state()
    st.markdown('<div class="hy-foot">HyFinEval · 犀牛鸟开源实战任务个人作品 · '
                "模型：腾讯混元 Hy3 / HunYuan</div>", unsafe_allow_html=True)
    st.stop()

sample = {"subtask": subtask, "input": question, "difficulty": "中"}
with st.spinner("Hy3 思考中…" if use_hy3 else "基线生成中…"):
    if use_hy3:
        out = hy3_app.generate(sample, use_rag=use_rag, pdf_pages=pdf_pages, pdf_name=pdf_name)
        if not out:
            st.warning("Hy3 调用未返回结果（可能未配置 key 或网络异常），已自动回退到基线示例。")
            out = baseline_app.baseline_generate(sample)
    else:
        out = baseline_app.baseline_generate(sample)

with st.expander("📝 输入", expanded=False):
    st.write(question)

if not out:
    st.error(
        "未获取到输出。使用 Hy3 时请确认：① 已设置环境变量 HY3_API_KEY；"
        "② HY3_BASE_URL 指向可用 endpoint（如 tokenhub-intl.tencentmaas.com/v1）。"
    )
    st.stop()

# 模型输出
st.markdown('<div class="hy-card"><h4>② 模型输出</h4></div>', unsafe_allow_html=True)
st.markdown(out.get("answer", ""))

# 引用 / 溯源
cites = out.get("citations") or []
st.markdown('<div class="hy-card" style="margin-top:16px;"><h4>③ 引用 / 溯源</h4></div>',
            unsafe_allow_html=True)
if cites:
    cit_html = ""
    for c in cites:
        if not isinstance(c, dict):
            continue
        if c.get("page") or c.get("source") in ("pdf", "report"):
            src = f"文档 · 第 {c.get('page', '?')} 页"
            field = c.get("excerpt") or c.get("field") or ""
        else:
            src = f"{data_store.company_name(c.get('company', ''))} {c.get('year', '')}"
            field = c.get("field") or ""
        val = c.get("value", "")
        cit_html += (
            '<div class="hy-cit">'
            f'<div class="src">{src}</div>'
            f'<div class="field">{field}</div>'
            f'<div class="val">{val}</div>'
            "</div>"
        )
    st.markdown(cit_html, unsafe_allow_html=True)
else:
    st.info("本次输出未携带结构化 citations（可能为闭卷或基线模式）。")

# 实时评估：复用同一套 7 维 rubric，离线可跑
res = evaluator.evaluate(sample, out)
dims = res["dimensions"]
overall = res["overall"]
breaker = res.get("compliance_breaker")

# 综合分卡片
ov_col = _score_color(overall / 100.0)
st.markdown(
    '<div class="hy-card"><div class="hy-score-wrap">'
    f'<div class="hy-score" style="color:{ov_col}">{overall}'
    '<span class="u">/100</span></div>'
    "<div>"
    '<div class="hy-score-name">④ 综合评分</div>'
    f'<div class="hy-fm">失败模式：{res["failure_mode"]}</div>'
    "</div></div></div>",
    unsafe_allow_html=True,
)
if breaker:
    st.error(f"⚠️ 合规熔断触发：{breaker}（综合分已封顶）")

# 七维明细卡片
dim_html = ""
for d, w in config.DIMENSION_WEIGHTS.items():
    v = dims.get(d, 0.0)
    col = _score_color(v)
    pct = round(v * 100, 1)
    dim_html += (
        '<div class="hy-dim"><div class="lab">'
        f"<span>{config.DIMENSION_NAMES[d]}</span>"
        f'<span class="wt">权重 {w} · {v:.2f}</span></div>'
        f'<div class="hy-bar"><span style="width:{pct}%;background:{col}"></span></div>'
        "</div>"
    )
st.markdown(f'<div class="hy-card"><h4>七维评分明细</h4>{dim_html}</div>',
            unsafe_allow_html=True)

st.markdown('<div class="hy-foot">HyFinEval · 犀牛鸟开源实战任务个人作品 · '
            "模型：腾讯混元 Hy3 / HunYuan</div>", unsafe_allow_html=True)
