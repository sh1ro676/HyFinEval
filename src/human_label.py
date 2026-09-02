# -*- coding: utf-8 -*-
"""人工标注 UI（B 组 · 独立页面，轻量标注版）。

改为「非专家也能标」的轻量方案：不要求打 0–100 绝对分（那需要金融功底），
只让标注者做两件【可观测】的事：
  1) 整体质量三档（优 / 中 / 差）——基于可读性、有无明显问题判断，无需领域知识；
  2) 若干可观测勾选项（格式完整 / 引用可溯源 / 无合规红线 / 无自相矛盾硬伤）——
     这些信号看得到就能勾，不需要判断「对不对」。

用于获取独立于自动 rubric 的人工判定，从而计算 auto-vs-human 一致性，
以及（两位标注者各标一遍）标注者间 Cohen's Kappa。

运行：  streamlit run src/human_label.py
数据：  读取 data_cache/label_pool.json（由 build_label_pool.py 生成）
保存：  data_cache/human_labels.json（{id: {annotator: {band, flags}}})
"""
import json
import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_store
import config

POOL = os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json")
LAB = os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json")

BANDS = ["优", "中", "差"]
BAND_HELP = {
    "优": "可读、结构完整、无明显错误/红线",
    "中": "能用，但存在瑕疵（如缺小节、引用不全、轻微问题）",
    "差": "明显错误、自相矛盾、触发合规红线或基本不可用",
}
# 可观测勾选项：True = 该正向质量信号成立（看得到就勾，不需要判断事实对错）
FLAGS = [
    ("format_ok", "格式完整（含结论 / 关键指标 / 风险与关注 / 简要分析 四小节）"),
    ("cited", "引用可溯源（出现的数字基本都带引用行）"),
    ("no_redline", "无合规红线（未出现荐股 / 保本 / 目标价等话术）"),
    ("no_contra", "无明显的自相矛盾或事实硬伤"),
]

st.set_page_config(page_title="HyFinEval · 人工标注", layout="wide")
st.title("HyFinEval 人工质量标注（轻量盲标）")
st.caption("无需金融功底：只看『整体三档 + 可观测勾选』。请依据输出本身质量判断，不要被题目难度干扰。")


def load_pool():
    if not os.path.exists(POOL):
        st.error("未找到标注池，请先运行 `python src/build_label_pool.py`")
        return []
    return json.load(open(POOL, encoding="utf-8"))


def load_labels():
    if os.path.exists(LAB):
        try:
            return json.load(open(LAB, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _valid(entry):
    """某标注者的值是否为新 schema（含 band）。旧占位 int 视为无效。"""
    return isinstance(entry, dict) and "band" in entry


pool = load_pool()
labels = load_labels()

if not pool:
    st.stop()

annotator = st.sidebar.selectbox("标注者", ["A", "B"], help="两位标注者各用 A/B 跑一遍，可算 Kappa")
show_auto = st.sidebar.checkbox("显示自动分（仅校对用，正式标注请关闭）", value=False)
st.sidebar.markdown("---")
st.sidebar.info("整体档：优=无明显问题；中=有瑕疵但能用；差=明显错误 / 红线 / 不可用。")

idx = st.sidebar.slider("样本序号", 0, len(pool) - 1, 0)
item = pool[idx]
item_id = item["id"]

saved = labels.get(item_id, {}).get(annotator) if isinstance(labels.get(item_id), dict) else None
saved = saved if _valid(saved) else None
default_band = saved["band"] if saved else "中"
default_flags = saved.get("flags", {}) if saved else {}

st.subheader(f"样本 {item_id}  ·  子任务：{item.get('subtask')}  ·  构造难度：{item.get('difficulty')}")
st.markdown("**① 用户问题 / 输入**")
st.write(item.get("input"))
st.markdown("**② 模型输出**")


def _unescape_json_like(s):
    """把 JSON 字符串里的 \\n、\\t、\\" 等转义序列还原为真实字符。"""
    if not isinstance(s, str):
        return s
    # 顺序很重要：先处理 \\ 再处理 \"
    s = s.replace('\\\\', '\\')
    s = s.replace('\\n', '\n')
    s = s.replace('\\r', '\r')
    s = s.replace('\\t', '\t')
    s = s.replace('\\"', '"')
    s = s.replace("\\'", "'")
    return s


def _extract_answer_from_truncated_json(raw):
    """对截断/损坏的 JSON 字符串，尽最大努力提取 answer 字段文本。"""
    m = re.search(r'"answer"\s*:\s*"(.*)', raw, re.DOTALL)
    if not m:
        return ""
    txt = m.group(1)
    # 截断字符串常以未闭合的 " 结尾；尽量找到最后一个可闭合的位置
    # 策略：从末尾向前找未被转义的 "，若找不到则直接取到末尾并清理
    end = len(txt)
    for i in range(end - 1, -1, -1):
        if txt[i] == '"':
            # 检查前面连续奇数个反斜杠 -> 被转义，不算闭合
            backslashes = 0
            j = i - 1
            while j >= 0 and txt[j] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                end = i
                break
    txt = txt[:end]
    return _unescape_json_like(txt)


def _normalize_output(raw):
    """兼容旧版标注池：output 字段偶尔是 JSON 字符串（answer 里套了 JSON），
    此时提取内层 answer 文本，避免把 \\n/## 当纯文本显示。

    也兼容 JSON 被截断的情况（如模型输出过长被截断）。
    """
    if not raw:
        return "", []
    if isinstance(raw, dict):
        return raw.get("answer", ""), raw.get("citations", [])
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("answer", ""), parsed.get("citations", [])
        except Exception:
            pass
        # 解析失败：可能是截断 JSON，尝试提取 answer 文本
        answer_text = _extract_answer_from_truncated_json(raw)
        if answer_text:
            return answer_text, []
    return raw, []


answer_text, parsed_cits = _normalize_output(item.get("output"))
st.markdown(answer_text)
cits = item.get("citations") or parsed_cits or []
if cits:
    st.markdown("**③ 引用 / 溯源**")
    st.table([{"字段": c.get("field"), "数值": c.get("value"),
               "公司": c.get("company"), "年份": c.get("year")}
              for c in cits if isinstance(c, dict)])

if show_auto:
    st.warning(f"⚠️ 当前自动 rubric 分 = {item.get('auto_score')}，盲标时请勿参考")

band = st.radio("整体质量档", BANDS, index=BANDS.index(default_band),
                help="；".join(f"{k}：{v}" for k, v in BAND_HELP.items()))
st.markdown("**可观测勾选项**（看得到就勾，无需判断事实对错）")
flags = {}
for key, label in FLAGS:
    flags[key] = st.checkbox(label, value=bool(default_flags.get(key)))

if st.button("保存本条", type="primary"):
    rec = {"band": band, "flags": flags}
    labels.setdefault(item_id, {})[annotator] = rec
    json.dump(labels, open(LAB, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    st.success(f"已保存 {item_id} 标注者 {annotator}：{band} · 勾选 {sum(flags.values())}/4")


def _count(aid):
    n = 0
    for v in labels.values():
        if isinstance(v, dict) and _valid(v.get(aid)):
            n += 1
    return n


done_a, done_b = _count("A"), _count("B")
st.sidebar.markdown(f"进度：A 已标 {done_a}/{len(pool)}，B 已标 {done_b}/{len(pool)}")
st.sidebar.markdown("---")
st.sidebar.info("第二位标注者把上方『标注者』切到 B 再标一遍同一批，即可在 label_stats.py 出 Kappa。")
if os.path.exists(LAB):
    st.sidebar.download_button("下载 human_labels.json",
                               open(LAB, "r", encoding="utf-8").read(),
                               file_name="human_labels.json")
