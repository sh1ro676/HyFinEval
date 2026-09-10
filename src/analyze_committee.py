# -*- coding: utf-8 -*-
"""裁判委员会分歧度信号验证。

核心假设：裁判分歧度（committee_std）是自动评分不确定性的有效信号——
分歧小的样本，自动分与人类对齐度更高；分歧大的样本，对齐度更低。

用法：
  python src/analyze_committee.py
  python src/analyze_committee.py --committee src/results/committee_cv.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_utils


def load_json(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path) if not os.path.isabs(path) else path
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def band_to_score(band: str) -> int:
    return {"优": 90, "中": 60, "差": 30}.get(band, 60)


def band_to_idx(band: str) -> int:
    return {"优": 2, "中": 1, "差": 0}.get(band, 1)


def compute_group_metrics(rule_scores, human_scores, human_bands):
    """计算一组样本的 κ、Spearman、MAE、一致率。"""
    n = len(rule_scores)
    if n < 2:
        return None
    # Spearman
    rho = stats_utils.spearman(rule_scores, human_scores)
    # MAE
    mae = sum(abs(rule_scores[i] - human_scores[i]) for i in range(n)) / n
    # 三档一致率
    auto_bands = []
    for s in rule_scores:
        if s >= 75:
            auto_bands.append("优")
        elif s >= 45:
            auto_bands.append("中")
        else:
            auto_bands.append("差")
    strict = sum(1 for i in range(n) if auto_bands[i] == human_bands[i]) / n
    adjacent = sum(1 for i in range(n)
                   if abs(band_to_idx(auto_bands[i]) - band_to_idx(human_bands[i])) <= 1) / n
    # 二次加权 κ（三档 0/1/2）
    auto_idx = [band_to_idx(b) for b in auto_bands]
    hum_idx = [band_to_idx(b) for b in human_bands]
    kappa = stats_utils.quad_weighted_kappa(auto_idx, hum_idx, k=3)
    return {
        "n": n,
        "spearman": round(rho, 3) if rho is not None else None,
        "kappa": round(kappa, 3) if kappa is not None else None,
        "mae": round(mae, 2),
        "strict_agreement": round(strict, 3),
        "adjacent_agreement": round(adjacent, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committee", default="src/results/committee_cv.json")
    ap.add_argument("--human", default="data_cache/human_labels.json")
    ap.add_argument("--out", default="src/results/committee_analysis.json")
    args = ap.parse_args()

    cv = load_json(args.committee)
    human = load_json(args.human)

    # 构建对齐数据：只保留同时有委员会分、规则分、人工标注的样本
    aligned = []
    for r in cv.get("samples", []):
        sid = r["id"]
        h = human.get(sid, {})
        # 取标注者 A（单人标注）
        ann = h.get("A") or h.get("B") or next(iter(h.values()), None) if isinstance(h, dict) else None
        if not ann:
            continue
        c = r.get("committee")
        if not c:
            continue
        aligned.append({
            "id": sid,
            "rule_score": r["rule_score"],
            "committee_mean": c["committee_mean"],
            "committee_std": c["committee_std"],
            "committee_reliability": c["committee_reliability"],
            "human_band": ann.get("band", "中"),
            "human_score": band_to_score(ann.get("band", "中")),
        })

    n_total = len(aligned)
    if n_total < 6:
        print(f"对齐样本仅 {n_total} 条，不足以做分组验证。需确保 committee_cv.json 已生成且 human_labels.json 非空。")
        return

    # 全量指标
    rule_all = [a["rule_score"] for a in aligned]
    hum_all = [a["human_score"] for a in aligned]
    band_all = [a["human_band"] for a in aligned]
    all_metrics = compute_group_metrics(rule_all, hum_all, band_all)
    all_metrics["description"] = "全量样本"

    # 按分歧度中位数分组
    stds = [a["committee_std"] for a in aligned]
    median_std = sorted(stds)[len(stds) // 2]
    low = [a for a in aligned if a["committee_std"] < median_std]
    high = [a for a in aligned if a["committee_std"] >= median_std]

    low_metrics = compute_group_metrics(
        [a["rule_score"] for a in low],
        [a["human_score"] for a in low],
        [a["human_band"] for a in low],
    )
    low_metrics["description"] = f"低分歧组（std < {median_std}）"

    high_metrics = compute_group_metrics(
        [a["rule_score"] for a in high],
        [a["human_score"] for a in high],
        [a["human_band"] for a in high],
    )
    high_metrics["description"] = f"高分歧组（std >= {median_std}）"

    # 按可靠性标签分组（三档）
    rel_groups = {}
    for rel in ["可信", "存疑", "需人工复核"]:
        g = [a for a in aligned if a["committee_reliability"] == rel]
        if len(g) >= 3:
            rel_metrics = compute_group_metrics(
                [a["rule_score"] for a in g],
                [a["human_score"] for a in g],
                [a["human_band"] for a in g],
            )
            rel_metrics["description"] = f"可靠性={rel}"
            rel_groups[rel] = rel_metrics

    # 子任务级诊断
    subtask_groups = {}
    # 需要从 label_pool 取 subtask 信息
    try:
        pool = load_json("data_cache/label_pool.json")
        id_to_subtask = {}
        for item in pool:
            id_to_subtask[item.get("id")] = item.get("subtask", "")
        for a in aligned:
            a["subtask"] = id_to_subtask.get(a["id"], "")
        from collections import defaultdict
        by_sub = defaultdict(list)
        for a in aligned:
            by_sub[a["subtask"]].append(a)
        for st, g in by_sub.items():
            if len(g) >= 3:
                m = compute_group_metrics(
                    [x["rule_score"] for x in g],
                    [x["human_score"] for x in g],
                    [x["human_band"] for x in g],
                )
                m["description"] = f"子任务={st}"
                subtask_groups[st] = m
    except Exception:
        pass

    report = {
        "n_total": n_total,
        "median_std": round(median_std, 2),
        "all": all_metrics,
        "low_disagreement": low_metrics,
        "high_disagreement": high_metrics,
        "by_reliability": rel_groups,
        "by_subtask": subtask_groups,
        "samples": aligned,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("裁判委员会分歧度信号验证报告")
    print("=" * 60)
    print(f"对齐样本数 n = {n_total} | 分歧度中位数 = {median_std}")
    print()
    for key, label in [("all", "全量"), ("low_disagreement", "低分歧组"), ("high_disagreement", "高分歧组")]:
        m = report[key]
        print(f"【{label}】 n={m['n']} | κ={m['kappa']} | ρ={m['spearman']} | "
              f"MAE={m['mae']} | 严格一致={m['strict_agreement']} | 相邻一致={m['adjacent_agreement']}")
    print()
    if rel_groups:
        print("按可靠性标签：")
        for rel, m in rel_groups.items():
            print(f"  {rel}: n={m['n']} κ={m['kappa']} ρ={m['spearman']} MAE={m['mae']}")
    if subtask_groups:
        print()
        print("按子任务：")
        for st, m in subtask_groups.items():
            print(f"  {st}: n={m['n']} κ={m['kappa']} ρ={m['spearman']} MAE={m['mae']}")
    print()
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()
