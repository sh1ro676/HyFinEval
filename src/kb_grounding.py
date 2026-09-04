# -*- coding: utf-8 -*-
"""KB-grounding 最小验证模块（仅用于 公告摘要 子任务）。

当前评估器对「公告摘要」退化成关键词启发式：
  factual_accuracy = 0.9 if "以原文为准" in text else 0.6   (零方差)
  citation_verifiability = 1.0 if "以原文为准" in text else ...  (96% 满分)
导致该子任务与人工档位 Spearman = -0.466（方向全反）。

本模块把 data_cache/announcements.json（元数据级 KB：code/title/type/time）
当作"真实公告类型"真值，做两件事：
  1. grounded_factual：核对模型自报的公告类型是否与 KB 真值一致；
  2. grounded_citation：检测模型是否编造了具体金额/比例（题面要求未知金额"以原文为准"）。
从而把"无真值"的子任务变成"可查表验证"的子任务，验证「扩充 KB 能否解决失效」这一假设。

注意：announcements.json 只含元数据，不含公告正文，因此本模块只验证
「公告类型」一项真值；全文 faithfulness 需要正文语料，属后续扩展。
"""
import os
import re
import json
import config

_ANN = None


def load_announcements():
    global _ANN
    if _ANN is None:
        path = os.path.join(config.ROOT_DIR, "data_cache", "announcements.json")
        with open(path, encoding="utf-8") as f:
            _ANN = json.load(f)
    return _ANN


def _parse_input_meta(inp):
    """从样本 input 解析 (code, title, date)。

    title：取『公告标题：』之后、到首个『（』或首个句末『。请』为止；
    兼容标题后无括号的情况（如『…公告。请生成要点摘要。』）。
    """
    code = None
    m = re.search(r"\((\d{4,6})", inp)
    if m:
        code = m.group(1)
    title = ""
    m = re.search(r"公告标题：\s*(.+?)\s*(?:[（(]|。请|。$)", inp)
    if m:
        title = m.group(1).strip()
    date = None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", inp)
    if m:
        date = m.group(1)
    return code, title, date


def match_announcement(inp):
    """按 标题子串（+code 校验）匹配 KB 中的真实公告，返回 {true_type, matched}。"""
    code, title, date = _parse_input_meta(inp)
    best = None
    for a in load_announcements():
        if code and a.get("code") != code:
            continue
        at = a.get("title", "")
        if title and (title in at or at in title):
            best = a
            break
    if best is None:
        return {"true_type": None, "matched": False}
    return {"true_type": best.get("type"), "matched": True}


# ---- 公告类型语义归一 & 同义组 ----
_GROUP_KEYWORDS = {
    "分红": ["分红", "利润分配", "派息", "派发", "派送"],
    "增持": ["增持", "股东增持", "高管增持"],
    "回购": ["回购", "股份回购"],
    "股权激励": ["股权激励", "期权", "限制性股票", "激励"],
    "业绩": ["业绩", "说明会", "预告", "快报"],
}


def _groups(t):
    s = str(t)
    return {g for g, kws in _GROUP_KEYWORDS.items() if any(k in s for k in kws)}


def _normalize(t):
    return re.sub(r"[\s/类公告]",
                  "",
                  str(t))


def type_match(claimed, true_type):
    """claimed 与 true_type 是否语义一致（同义组或包含）。"""
    if not claimed or not true_type:
        return False
    c, t = _normalize(claimed), _normalize(true_type)
    if c and t and (c in t or t in c):
        return True
    gc, gt = _groups(claimed), _groups(true_type)
    return bool(gc & gt)


# 规范类目 -> 命中关键词（按特异性从高到低；命中即认为模型自报该类目）
_CANON = [
    ("利润分配/分红", ["利润分配", "分红", "派息", "特别分红", "现金分红"]),
    ("股东/高管增持", ["增持", "股东增持", "高管增持"]),
    ("股份回购", ["股份回购", "回购公司股份", "回购"]),
    ("股权激励", ["股权激励", "股票期权", "限制性股票", "期权注销", "限制性股票激励计划"]),
    ("业绩预告/快报", ["业绩说明会", "业绩预告", "业绩快报", "说明会"]),
]


def extract_claimed_type(text):
    """从模型输出抽取其自报的公告类型（优先显式陈述，避免全文误命中）。

    模型输出是自由文本，常见显式模式：
      - 『公告类型：XXX』 / 『公告类型：XXX（…）』
      - 『属于…公告』 / 『该公告为…的披露文件』
    优先从这些显式陈述里取类型词；找不到再退回扫描『结论』段（前 250 字）。
    均失败返回 None（视为模型未给出类型判断）。
    """
    # 1) 显式：公告类型：XXX
    m = re.search(r"公告类型[：:]\s*([\u4e00-\u9fa5（）()/、]{1,16})", text)
    if m:
        return _canon_of(m.group(1))
    # 2) 显式：属于/为 … 公告（截到 公告/类别/披露文件/类）
    m = re.search(r"(?:属于|为|是)\s*([^，。\n]{1,18}?)\s*(?:公告|类别|披露文件|类)", text)
    if m:
        c = _canon_of(m.group(1))
        if c:
            return c
    # 3) 退回：结论段（前 250 字）扫描规范类目关键词
    head = text[:250]
    for canon, kws in _CANON:
        for kw in kws:
            if kw in head:
                return canon
    return None


def _canon_of(phrase):
    """把任意短语映射到规范类目（命中即返回），否则 None。"""
    for canon, kws in _CANON:
        for kw in kws:
            if kw in phrase:
                return canon
    return None


_NUM_PAT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:亿|万|万元|亿元|元|万股|万份|%|\百分比)"
    r"|\d+\.\d+\s*%"
)


def grounded_factual(output, true_type):
    """公告类型核对：命中真值=1.0；模型自称无法判断但真值存在=0.3；错判=0.0。"""
    claimed = extract_claimed_type(output)
    if not true_type:
        return 0.6  # 无 KB 真值，保守中性
    if claimed is None:
        # 没给出类型判断：若明确说"无法判断/未知"给中性，否则偏低
        if any(w in output for w in ("无法判断", "未知类型", "难以判定", "不确定类型")):
            return 0.3
        return 0.4
    return 1.0 if type_match(claimed, true_type) else 0.0


def grounded_citation(output):
    """编造具体金额/比例检测：题面要求未知金额『以原文为准』。

    若输出断言了具体金额/比例，且同一句无『以原文为准/未披露/待公告』等
    免责锚点 → 视为不可验证的编造，降分。
    """
    for m in _NUM_PAT.finditer(output):
        seg = output[max(0, m.start() - 40): m.end() + 10]
        if any(w in seg for w in ("以原文为准", "未披露", "待公告", "未知", "具体数字须", "须以")):
            continue
        return 0.3  # 出现无锚点硬数字 → 不可验证
    return 1.0
