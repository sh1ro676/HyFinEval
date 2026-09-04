# -*- coding: utf-8 -*-
"""基于真实公告原文，重建「公告摘要」样本集（开卷 + 闭卷配对）。

为什么重建
----------
原 16 条公告摘要样本只有巨潮**标题**、没有正文，模型只能闭卷输出"该类公告通常
应包含…以原文为准"的通用框架。实测这类输出两两相似度仅 0.059（并不雷同），但
缺乏**可判别的信息增量差异**——人工在同一套模板上分优/中/差，评的其实是任务设置
而非模型质量，标注效度存疑（README E.6）。

本脚本用真实原文重建该子任务，并解决三件事：
  1. **开卷**：把公告正文喂给模型，输出才有"干货"，才可能在忠实度/覆盖度上拉开差距。
  2. **配对对照**：同一篇公告生成开卷/闭卷两条样本，构成"同题 × 异上下文"的干净对照，
     补上 README 中"开闭卷非干净对照"的缺口。
  3. **客观真值**：从喂给模型的原文中自动抽取关键数值（金额/比例/日期/股数），
     作为 ground-truth facts，供后续自动计算 fact_coverage 与幻觉率——
     这类指标不依赖主观标注，直接缓解"缺真值"问题。

用法
----
    python src/build_announcement_samples.py              # 生成 samples + 真值
    python src/build_announcement_samples.py --limit 24   # 限制公告篇数

输出
----
    data_cache/ann_samples.json       开卷/闭卷配对样本（含 ground_truth_facts）
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXTS = os.path.join(ROOT, "data_cache", "announcement_texts.json")
OUT = os.path.join(ROOT, "data_cache", "ann_samples.json")
ID_START = 105                      # 现有 samples.json 最大为 FIN-104
CONTEXT_LIMIT = 5000                # 喂给模型的原文上限（字符）

# ---- 关键事实抽取模式（ground-truth facts）----
MONEY = re.compile(r"\d[\d,]*\.?\d*\s*(?:亿元|万元|元|股|份|手|万股|亿股)")
RATIO = re.compile(r"\d+(?:\.\d+)?\s*%")
DATE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}")
YMD_DASH = re.compile(r"\d{4}/\d{1,2}/\d{1,2}")


def normalize(s):
    """归一化数值串：去掉空白与千分位，便于后续与模型输出做匹配。"""
    return re.sub(r"[\s,]", "", s or "")


def extract_facts(text):
    """从公告正文抽取关键事实（金额/比例/日期/股数），去重后返回。

    只从**实际喂给模型的文本**中抽取，保证 coverage 指标对模型公平——
    不能要求模型覆盖它根本没看到的内容。
    """
    facts = {"amount": [], "ratio": [], "date": []}
    for m in MONEY.findall(text):
        v = normalize(m)
        if len(v) >= 2 and v not in facts["amount"]:
            facts["amount"].append(v)
    for m in RATIO.findall(text):
        v = normalize(m)
        if v not in facts["ratio"]:
            facts["ratio"].append(v)
    for m in DATE.findall(text):
        v = normalize(m).replace("年", "-").replace("月", "-").rstrip("-")
        if v not in facts["date"]:
            facts["date"].append(v)
    for m in YMD_DASH.findall(text):
        if m not in facts["date"]:
            facts["date"].append(m)
    return facts


def select_context(text, limit=CONTEXT_LIMIT):
    """截断正文：保头优先（公告要点集中在开头的"重要内容提示"与前几节）。"""
    text = text.strip()
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    cut = head.rfind("\n")
    return (head[:cut] if cut > limit * 0.6 else head), True


def build_open(rec, ctx, truncated):
    """开卷样本：提供原文，要求基于原文摘要。"""
    inp = (
        "请阅读以下公告原文，完成摘要任务。\n\n"
        "【公告标题】{title}\n"
        "【公司】{name}（{code}）\n"
        "【披露日期】{date}\n\n"
        "【公告原文】\n{ctx}\n\n"
        "请输出：\n"
        "1）公告类型判断；\n"
        "2）该公告的关键要素，必须包含原文中出现的具体数值、比例、日期、涉及主体；\n"
        "3）对投资者而言的要点。\n"
        "要求：所有信息必须来自上述原文，不得引入原文之外的信息；"
        "原文未提及的内容请明确注明「原文未披露」，不得编造。"
    ).format(title=rec["title"], name=rec["name"], code=rec["code"],
             date=rec["time"], ctx=ctx)
    return inp


def build_closed(rec):
    """闭卷样本：只给标题（等同原设置），作为对照。"""
    return (
        "公告标题：{title}（{name}，{code}，{date}）。请判断该公告类型，"
        "并列出该类公告通常应包含的关键要素；对未知的具体金额/比例请注明"
        "\"以公告原文为准\"，不得编造。"
    ).format(title=rec["title"], name=rec["name"], code=rec["code"],
             date=rec["time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多用几篇公告（0=全部）")
    ap.add_argument("--min-facts", type=int, default=3,
                    help="关键事实少于此数的公告丢弃（信息量不足，无法衡量覆盖度）")
    args = ap.parse_args()

    if not os.path.exists(TEXTS):
        print("未找到 %s，请先运行 src/fetch_announcements.py" % TEXTS)
        return
    recs = json.load(open(TEXTS, encoding="utf-8"))
    if args.limit:
        recs = recs[: args.limit]
    print("载入公告原文 %d 篇" % len(recs))

    samples, skipped = [], 0
    nid = ID_START
    for rec in recs:
        ctx, truncated = select_context(rec.get("text", ""))
        facts = extract_facts(ctx)
        n_facts = len(facts["amount"]) + len(facts["ratio"]) + len(facts["date"])
        if n_facts < args.min_facts:
            skipped += 1
            continue
        base = {
            "code": rec["code"], "name": rec["name"], "title": rec["title"],
            "type": rec["type"], "time": rec["time"], "url": rec["url"],
            "n_facts": n_facts, "truncated": truncated,
            "ground_truth_facts": facts,
            # 原文随样本保存：开卷时它就是喂给模型的上下文；闭卷时模型看不到它，
            # 但标注者需要它来核对事实（否则闭卷输出无从判断对错）。
            "context_text": ctx,
        }
        # 开卷 / 闭卷配对：同一篇公告两种上下文，构成干净对照
        open_id, closed_id = "FIN-%03d" % nid, "FIN-%03d" % (nid + 1)
        nid += 2
        samples.append(dict(base, **{
            "id": open_id, "subtask": "公告摘要", "difficulty": "中",
            "input": build_open(rec, ctx, truncated),
            "mode": "开卷", "pair_id": closed_id,
            "source": "巨潮公告原文(真实PDF)", "is_counterfeit": False,
            "notes": "开卷：提供公告正文，要求基于原文摘要",
        }))
        samples.append(dict(base, **{
            "id": closed_id, "subtask": "公告摘要", "difficulty": "中",
            "input": build_closed(rec),
            "mode": "闭卷", "pair_id": open_id,
            "source": "巨潮披露列表(仅标题)", "is_counterfeit": False,
            "notes": "闭卷：仅标题，作为开卷对照",
        }))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    n_pair = len(samples) // 2
    print("生成样本 %d 条（开卷/闭卷配对 %d 组），跳过 %d 篇（关键事实<%d）"
          % (len(samples), n_pair, skipped, args.min_facts))
    print("id 范围：%s ~ %s" % (samples[0]["id"], samples[-1]["id"]))
    print("输出 -> %s" % OUT)

    # 类型分布
    dist = {}
    for s in samples:
        if s["mode"] == "开卷":
            dist[s["type"]] = dist.get(s["type"], 0) + 1
    print("公告类型分布：")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print("  %-14s %d 篇" % (k, v))
    nf = [s["n_facts"] for s in samples if s["mode"] == "开卷"]
    if nf:
        print("每篇关键事实数：均值 %.1f  最少 %d  最多 %d"
              % (sum(nf) / len(nf), min(nf), max(nf)))


if __name__ == "__main__":
    main()
