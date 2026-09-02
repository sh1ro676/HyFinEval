"""人工标注 UI（B 组 · 独立页面）。

用于获取「独立于自动 rubric 的人工质量判定」，从而合法计算
auto-vs-human Spearman 与 标注者间 Cohen's Kappa。

运行：  streamlit run src/human_label.py
数据：  读取 data_cache/label_pool.json（由 build_label_pool.py 生成）
保存：  data_cache/human_labels.json（{id: {annotator: score}}）

标注者请【盲标】：本页默认隐藏 auto_score，仅展示 输入/模型输出/引用，
请依据「输出质量」而非「题目难度」打 0–100 分。
"""
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_store
import config

POOL = os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json")
LAB = os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json")

st.set_page_config(page_title="HyFinEval · 人工标注", layout="wide")
st.title("HyFinEval 人工质量标注（盲标）")

annotator = st.sidebar.selectbox("标注者", ["A", "B"], help="两位标注者各用 A/B 跑一遍，可算 Kappa")
show_auto = st.sidebar.checkbox("显示自动分（仅校对用，正式标注请关闭）", value=False)
st.sidebar.markdown("---")
st.sidebar.info("请依据【输出质量】打分，不要被题目难度干扰。0–100，越高越好。")


def load_pool():
    if not os.path.exists(POOL):
        st.error("未找到标注池，请先运行 `python src/build_label_pool.py`")
        return []
    return json.load(open(POOL, encoding="utf-8"))


def load_labels():
    if os.path.exists(LAB):
        return json.load(open(LAB, encoding="utf-8"))
    return {}


pool = load_pool()
labels = load_labels()

if not pool:
    st.stop()

idx = st.sidebar.slider("样本序号", 0, len(pool) - 1, 0)
item = pool[idx]
item_id = item["id"]

# 已存分数
saved = labels.get(item_id, {}).get(annotator)
default = int(saved) if saved is not None else 70

st.subheader(f"样本 {item_id}  ·  子任务：{item.get('subtask')}  ·  构造难度：{item.get('difficulty')}")
st.markdown("**① 用户问题 / 输入**")
st.write(item.get("input"))
st.markdown("**② 模型输出**")
st.write(item.get("output"))
cits = item.get("citations") or []
if cits:
    st.markdown("**③ 引用 / 溯源**")
    st.table([{"字段": c.get("field"), "数值": c.get("value"),
               "公司": c.get("company"), "年份": c.get("year")}
              for c in cits if isinstance(c, dict)])

if show_auto:
    st.warning(f"⚠️ 当前自动 rubric 分 = {item.get('auto_score')}，盲标时请勿参考")

score = st.slider("人工质量分（0–100）", 0, 100, default)

if st.button("保存本条", type="primary"):
    labels.setdefault(item_id, {})[annotator] = score
    json.dump(labels, open(LAB, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    st.success(f"已保存 {item_id} 标注者 {annotator} = {score}")

# 进度
done_a = sum(1 for v in labels.values() if "A" in v)
done_b = sum(1 for v in labels.values() if "B" in v)
st.sidebar.markdown(f"进度：A 已标 {done_a}/{len(pool)}，B 已标 {done_b}/{len(pool)}")
if os.path.exists(LAB):
    st.sidebar.download_button("下载 human_labels.json",
                               open(LAB, "r", encoding="utf-8").read(),
                               file_name="human_labels.json")
