# -*- coding: utf-8 -*-
"""公告摘要的**客观**评分器：基于原文真值，不依赖主观判断。

设计动机
--------
原评估器在公告摘要上是启发式的（"以原文为准"→0.9），与人工 Spearman 为 −0.466，
且 factual 维度零方差。根因是**没有可核验的真值**。

喂了公告原文之后，真值变得可计算：原文里的金额/比例/日期就是客观事实集。
于是可以定义两个硬指标：

  fact_coverage      模型输出覆盖了原文关键事实的比例（越高越好）
  hallucination_rate 模型输出中的数值有多少在原文里找不到（越低越好）

这两者是**客观数值匹配**，不依赖任何主观判断，也不需要人工标注，
因此解决了该子任务"缺真值"的问题，并为后续人工标注提供可对照的客观锚点。

用法
----
    python src/score_announcement.py
    python src/score_announcement.py --show 3      # 打印 3 条明细

输出
----
    data_cache/ann_scores.json
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from ann_utils import MONEY, RATIO, DATE, YMD_DASH, norm  # noqa: E402

OUTS = os.path.join(ROOT, "data_cache", "ann_outputs.json")
OUT = os.path.join(ROOT, "data_cache", "ann_scores.json")

HEDGE = ["以原文为准", "以公告原文", "据公开知识", "未经", "未核实",
         "不得编造", "需以", "原文未披露", "无法确认"]


def num_core(s):
    """取归一化后的数值核心（去千分位与空白，保留小数）。"""
    m = re.search(r"\d+(?:\.\d+)?", norm(s))
    return m.group(0) if m else None


def extract_nums(text, with_units=True):
    """抽取文本中的全部数值（金额/比例/日期），返回数值核心集合。"""
    text = text or ""
    vals = set()
    for m in MONEY.findall(text):
        c = num_core(m)
        if c:
            vals.add(c)
    for m in RATIO.findall(text):
        c = num_core(m)
        if c:
            vals.add(c)
    for m in DATE.findall(text):
        v = norm(m).replace("年", "-").replace("月", "-").rstrip("-")
        parts = [p for p in re.split(r"[-/]", v) if p]
        if len(parts) >= 3:
            vals.add("%s-%s-%s" % (parts[0], parts[1].zfill(2), parts[2].zfill(2)))
    for m in YMD_DASH.findall(text):
        p = m.split("/")
        if len(p) == 3:
            vals.add("%s-%s-%s" % (p[0], p[1].zfill(2), p[2].zfill(2)))
    return vals


def hedge_count(text):
    return sum((text or "").count(k) for k in HEDGE)


def substantive_lines(text):
    """实质行数：去掉小节标题、空行、免责套话后仍有内容的行。"""
    n = 0
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("#").strip()
        if len(s) < 6:
            continue
        core = s
        for k in HEDGE:
            core = core.replace(k, "")
        core = re.sub(r"[\s，。、：:；;（）()【】\[\]—\-]", "", core)
        if len(core) >= 6:
            n += 1
    return n


def score_one(rec):
    out_text = rec.get("output") or ""
    inp = rec.get("input") or ""
    gt = rec.get("ground_truth_facts") or {}

    # 上下文真值：开卷时 input 内含公告原文，闭卷时只有标题
    ctx_nums = extract_nums(inp) if rec.get("mode") == "开卷" else set()
    out_nums = extract_nums(out_text)

    # 1) fact_coverage：原文关键事实被覆盖的比例
    gt_all = []
    cov_by = {}
    for kind in ("amount", "ratio", "date"):
        vals = [num_core(v) for v in (gt.get(kind) or [])]
        vals = [v for v in vals if v]
        gt_all.extend(vals)
        hit = sum(1 for v in vals if v in out_nums)
        cov_by[kind] = (hit / len(vals)) if vals else None
    valid = [v for v in gt_all if v]
    fact_coverage = (sum(1 for v in valid if v in out_nums) / len(valid)) if valid else None

    # 2) 幻觉率：输出数值在原文中找不到的比例（仅开卷可算）
    if rec.get("mode") == "开卷" and out_nums:
        hallu = sum(1 for v in out_nums if v not in ctx_nums) / len(out_nums)
    else:
        hallu = None

    # 3) 闭卷违规率：闭卷下给出具体金额/比例数值的条数占比
    #    （闭卷要求"不得编造具体数字"，给了数字即为未标注来源的硬数值）
    if rec.get("mode") == "闭卷":
        hard = [v for v in out_nums if re.search(r"\d+\.\d", v) or len(v) >= 6]
        unsourced = (len(hard) > 0)
        unsourced_n = len(hard)
    else:
        unsourced, unsourced_n = None, None

    return {
        "id": rec["id"], "mode": rec["mode"], "type": rec.get("type"),
        "title": rec.get("title"), "pair_id": rec.get("pair_id"),
        "n_gt_facts": len(valid),
        "fact_coverage": fact_coverage,
        "coverage_amount": cov_by.get("amount"),
        "coverage_ratio": cov_by.get("ratio"),
        "coverage_date": cov_by.get("date"),
        "hallucination_rate": hallu,
        "n_out_nums": len(out_nums),
        "closed_unsourced": unsourced,
        "closed_unsourced_n": unsourced_n,
        "hedge_count": hedge_count(out_text),
        "substantive_lines": substantive_lines(out_text),
        "out_chars": len(out_text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(OUTS):
        print("未找到 %s，请先运行 src/gen_announcement_outputs.py" % OUTS)
        return
    with open(OUTS, encoding="utf-8") as _f:
        recs = json.load(_f)
    scores = [score_one(r) for r in recs]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=1)

    # 分组汇总
    def agg(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    print("=" * 68)
    print("客观指标汇总（开卷 vs 闭卷配对对照）")
    print("=" * 68)
    for mode in ("开卷", "闭卷"):
        rows = [s for s in scores if s["mode"] == mode]
        if not rows:
            continue
        print("\n【%s】n=%d" % (mode, len(rows)))
        for key, label in (("fact_coverage", "事实覆盖率"),
                           ("hallucination_rate", "幻觉率"),
                           ("hedge_count", "免责语次数"),
                           ("substantive_lines", "实质行数"),
                           ("out_chars", "输出字数")):
            m, n = agg(rows, key)
            if m is None:
                print("  %-10s —（该模式下不适用）" % label)
            else:
                print("  %-10s 均值 %.3f  (n=%d)" % (label, m, n))
        if mode == "闭卷":
            u = sum(1 for r in rows if r.get("closed_unsourced"))
            print("  给出了具体数值(未标注来源)的样本: %d/%d" % (u, len(rows)))

    # 配对差：同一篇公告，开卷 − 闭卷
    pairs = {}
    for s in scores:
        pairs.setdefault((s["type"], s["title"]), {})[s["mode"]] = s
    diffs = [v["开卷"]["fact_coverage"] - v["闭卷"]["fact_coverage"]
             for v in pairs.values()
             if "开卷" in v and "闭卷" in v
             and v["开卷"].get("fact_coverage") is not None
             and v["闭卷"].get("fact_coverage") is not None]
    if diffs:
        print("\n【配对差】开卷 fact_coverage − 闭卷 = 均值 %+.3f  (n=%d 对)"
              % (sum(diffs) / len(diffs), len(diffs)))
        print("  正值说明：喂原文后，输出对真实关键事实的覆盖显著提升。")

    if args.show:
        print("\n" + "-" * 68)
        for s in scores[: args.show]:
            print("[%s] %s %s" % (s["id"], s["mode"], (s["title"] or "")[:34]))
            print("  覆盖率 %.3f  幻觉率 %s  免责语 %d  实质行 %d"
                  % (s["fact_coverage"] if s["fact_coverage"] is not None else -1,
                     ("%.3f" % s["hallucination_rate"])
                     if s["hallucination_rate"] is not None else "—",
                     s["hedge_count"], s["substantive_lines"]))

    print("\n输出 -> %s" % OUT)


if __name__ == "__main__":
    main()
