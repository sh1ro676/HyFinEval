# -*- coding: utf-8 -*-
"""重建版「公告摘要」标注页（轻量盲标，三档 + 可观测勾选）。

启动：streamlit run src/human_label_ann.py

与 src/human_label.py 的区别
----------------------------
原标注页面向指标类样本；本页面向**喂了真实公告原文**的重建样本，因此：

1. 开卷样本：原文直接展示，标注者可判断"有没有说出该说的具体数字"。
   ——这正是原设置做不到的：原来只有标题，标注者无从判断信息量，只能凭
   "是不是车轱辘话"打档，标注效度存疑（README E.6）。
2. 闭卷样本：原文放在**折叠区**，明确标注"模型未看到"，仅供标注者核对事实。
   这样开卷/闭卷可在同一把尺子上比较，同时不破坏闭卷设置。
3. 勾选项换成对该子任务真正有判别力的四项（原四项在 28 条上 3 项 100% 命中、
   零方差，已无区分度）。
4. 新增「不确定 / 我没看懂」选项：允许标注者明确表达无法判断，避免把
   "看不懂"误记成"差"，分析时可剔除或做敏感性分析。

数据：读取 data_cache/ann_outputs.json → 保存 data_cache/human_labels_ann.json
"""
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

POOL = os.path.join(config.ROOT_DIR, "data_cache", "ann_outputs.json")
SAVE = os.path.join(config.ROOT_DIR, "data_cache", "human_labels_ann.json")
SCORES = os.path.join(config.ROOT_DIR, "data_cache", "ann_scores.json")

BANDS = ["优", "中", "差"]

# 可观测勾选项：针对公告摘要重新设计（原四项三项零方差，已换掉）
FLAGS = {
    "key_numbers_covered": "关键数值覆盖到位（金额/比例/日期说到了）",
    "no_fabrication": "没有编造原文中不存在的数字",
    "structured": "结构完整、有实质内容（非一两句话敷衍）",
    "directly_usable": "可直接采用（不必再去翻原文）",
}


@st.cache_data
def load_pool():
    if not os.path.exists(POOL):
        return None
    return json.load(open(POOL, encoding="utf-8"))


@st.cache_data
def load_scores():
    if not os.path.exists(SCORES):
        return {}
    return {s["id"]: s for s in json.load(open(SCORES, encoding="utf-8"))}


def load_labels():
    if os.path.exists(SAVE):
        return json.load(open(SAVE, encoding="utf-8"))
    return {}


def save_labels(labels):
    with open(SAVE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)


pool = load_pool()
if pool is None:
    st.error("未找到标注池，请先运行 `python src/gen_announcement_outputs.py`")
    st.stop()

labels = load_labels()
scores = load_scores()

st.set_page_config(page_title="HyFinEval · 公告摘要标注", layout="wide")
st.title("HyFinEval 公告摘要标注（重建版 · 开卷/闭卷配对）")
st.caption("开卷样本会展示公告原文：请重点判断输出有没有说出该说的具体数字。")

annotator = st.sidebar.selectbox("标注者", ["A", "B"])
mode_filter = st.sidebar.selectbox("只看", ["全部", "开卷", "闭卷"])
show_metrics = st.sidebar.checkbox("显示客观指标（正式标注请关闭）", value=False)
st.sidebar.markdown("---")
st.sidebar.info("优=信息齐全可直接用；中=有遗漏或冗余但能用；差=空壳/编造/不可用。")

view = [p for p in pool if mode_filter == "全部" or p.get("mode") == mode_filter]
if not view:
    st.warning("当前筛选下没有样本")
    st.stop()

idx = st.sidebar.slider("样本序号", 0, len(view) - 1, 0)
item = view[idx]
item_id = item["id"]
saved = labels.get(item_id, {}).get(annotator)
default_band = saved.get("band") if saved and "band" in saved else "中"
default_flags = saved.get("flags", {}) if saved else {}

done = sum(1 for v in labels.values() if annotator in v)
st.sidebar.markdown(f"进度（{annotator}）：{done}/{len(pool)}")
st.sidebar.progress(done / len(pool))

# ---- 样本展示 ----
st.subheader(f"{item_id} · {item.get('mode')} · {item.get('type')} · {item.get('name')}")
st.caption(f"公告标题：{item.get('title')}　披露日期：{item.get('time')}")

raw_input = item.get("input", "")
mode = item.get("mode")
if mode == "开卷":
    # 开卷：input 内含原文，直接展示（标注者与模型看到同样材料）
    st.markdown("**① 公告原文（模型可见）**")
    with st.expander("展开原文", expanded=False):
        st.text(raw_input[:6000])
else:
    # 闭卷：模型只看到标题；原文折叠，仅供标注者核对事实
    st.markdown("**① 模型看到的输入（仅标题）**")
    st.code(raw_input[:500], language="text")
    st.markdown("**①-b 公告原文（模型未看到，仅供你核对事实）**")
    with st.expander("展开原文（核对用）", expanded=False):
        st.text(item.get("_context_for_labeler", "（原文未随输出保存）")[:6000])

st.markdown("**② 模型输出**")
st.markdown(item.get("output", ""))

cits = item.get("citations") or []
if cits:
    st.markdown(f"**③ 引用（{len(cits)} 条）**")
    st.table([{"摘录": str(c.get("excerpt", ""))[:80]} for c in cits])

if show_metrics and item_id in scores:
    s = scores[item_id]
    cov = s.get("fact_coverage")
    hal = s.get("hallucination_rate")
    st.info(
        "客观指标：事实覆盖率 %s ｜ 幻觉率 %s ｜ 免责语 %d 次 ｜ 实质行 %d"
        % (("%.2f" % cov) if cov is not None else "—",
           ("%.2f" % hal) if hal is not None else "—",
           s.get("hedge_count", 0), s.get("substantive_lines", 0))
    )

st.markdown("---")
band = st.radio("整体质量档", BANDS, index=BANDS.index(default_band), horizontal=True)
st.markdown("**可观测勾选项**")
flags = {}
for key, label in FLAGS.items():
    flags[key] = st.checkbox(label, value=bool(default_flags.get(key)))
unsure = st.checkbox("⚠️ 不确定 / 我没看懂（分析时可剔除）",
                     value=bool(saved.get("unsure")) if saved else False)

if st.button("保存本条", type="primary"):
    labels.setdefault(item_id, {})
    labels[item_id][annotator] = {"band": band, "flags": flags, "unsure": unsure}
    save_labels(labels)
    st.success(f"已保存 {item_id}（{annotator}）：{band}")

if st.sidebar.download_button("下载标注结果", json.dumps(
        labels, ensure_ascii=False, indent=1), file_name="human_labels_ann.json"):
    pass
