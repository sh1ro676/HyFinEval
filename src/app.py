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

st.set_page_config(page_title="HyFinEval · 财报速读助手", layout="wide")
st.title("📊 HyFinEval · 财报速读助手")
st.caption(
    "基于腾讯混元 Hy3 的金融分析应用：指标提取 / 财务问答 / 公告摘要，输出带引用、可溯源；"
    "并实时给出 7 维评分。面向基金 / 行业分析师的「财报速读」真实用户场景。"
)

col_left, col_right = st.columns([1, 2])

with col_left:
    code = st.text_input("股票代码", value="600519", help="如 600519 贵州茅台")
    year = st.text_input("报告期（年）", value="2023")
    subtask = st.selectbox("任务类型", ["财报指标提取", "财务问答", "公告摘要"])
    use_rag = st.checkbox("开卷（RAG 注入真实指标表）", value=True)
    use_hy3 = st.checkbox("使用 Hy3 真实模型", value=config.USE_HY3,
                          help="取消勾选则使用基线示例（无需 API key）")
    if not config.USE_HY3:
        st.warning("⚠️ 未检测到 HY3_API_KEY 环境变量，将以基线示例运行（输出基于本地真实指标，无需联网）。"
                   "设置 key 后即可勾选调用真实 Hy3。")
    custom_q = st.text_area("自定义问题（留空则用模板）", value="", height=80)
    go = st.button("生成", type="primary")

if subtask == "财报指标提取":
    tmpl = (f"请从{data_store.company_name(code)}（{code}）{year}年财务数据中提取"
            f"【扣非净利润】的数值。")
elif subtask == "财务问答":
    tmpl = (f"请基于{data_store.company_name(code)}（{code}）{year}年财务数据，"
            f"分析其当年盈利能力与偿债压力。")
else:
    tmpl = (f"请摘要{data_store.company_name(code)}（{code}）{year}年相关公告的关键要素，"
            f"不编造具体数字。")
question = custom_q.strip() or tmpl

if go:
    sample = {"subtask": subtask, "input": question, "difficulty": "中"}
    with st.spinner("Hy3 思考中…" if use_hy3 else "基线生成中…"):
        if use_hy3:
            out = hy3_app.generate(sample, use_rag=use_rag)
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
                st.table([
                    {"字段": c.get("field"), "数值": c.get("value"),
                     "公司": c.get("company"), "年份": c.get("year")}
                    for c in cites if isinstance(c, dict)
                ])
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
