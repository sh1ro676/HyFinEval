#!/usr/bin/env python3
"""严格审计员全面打分结果分析。

- 28 条 label_pool：与人工标注对齐（κ / ρ / MAE / 一致率）
- 122 条公告摘要：与客观指标 fact_coverage 对齐（Spearman / Pearson / 分组对比）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_utils


def band_to_score(b):
    return {"优": 90, "中": 60, "差": 30}.get(b, 60)


def band_to_idx(b):
    return {"优": 2, "中": 1, "差": 0}.get(b, 1)


def compute_metrics(auto_scores, human_scores, human_bands):
    """算 κ / ρ / MAE / 严格一致 / 相邻一致。"""
    n = len(auto_scores)
    if n < 2:
        return {}
    auto_bands = ["优" if s >= 75 else "中" if s >= 45 else "差" for s in auto_scores]
    kappa = stats_utils.quad_weighted_kappa(
        [band_to_idx(b) for b in auto_bands],
        [band_to_idx(b) for b in human_bands],
        k=3,
    )
    rho = stats_utils.spearman(auto_scores, human_scores)
    mae = sum(abs(auto_scores[i] - human_scores[i]) for i in range(n)) / n
    strict = sum(1 for i in range(n) if auto_bands[i] == human_bands[i]) / n
    adjacent = sum(
        1 for i in range(n) if abs(band_to_idx(auto_bands[i]) - band_to_idx(human_bands[i])) <= 1
    ) / n
    return {
        "n": n,
        "kappa": round(kappa, 3) if kappa is not None else None,
        "spearman": round(rho, 3) if rho is not None else None,
        "mae": round(mae, 2),
        "strict_agree": round(strict, 3),
        "adjacent_agree": round(adjacent, 3),
    }


def main():
    # ---- 加载数据 ----
    with open("src/results/strict_auditor_all.json", encoding="utf-8") as f:
        sa = json.load(f)
    with open("data_cache/human_labels.json", encoding="utf-8") as f:
        human = json.load(f)
    with open("data_cache/ann_scores.json", encoding="utf-8") as f:
        ann_scores = json.load(f)

    # 建立快速查找
    sa_by_id = {r["id"]: r for r in sa["samples"] if r["status"] == "ok"}
    ann_by_id = {s["id"]: s for s in ann_scores}

    # ========== 第一部分：28 条 label_pool vs 人工标注 ==========
    auto_scores, human_scores, human_bands = [], [], []
    for sid, h in human.items():
        if sid not in sa_by_id:
            continue
        ann = h.get("A") or h.get("B") or next(iter(h.values()), None) if isinstance(h, dict) else None
        if not ann:
            continue
        auto_scores.append(sa_by_id[sid]["overall"])
        human_scores.append(band_to_score(ann.get("band", "中")))
        human_bands.append(ann.get("band", "中"))

    label_metrics = compute_metrics(auto_scores, human_scores, human_bands)

    # ========== 第二部分：122 条公告摘要 vs fact_coverage ==========
    ann_judge_scores, ann_coverage = [], []
    for sid, r in sa_by_id.items():
        if sid.startswith("FIN-") and sid not in human:
            # 公告摘要样本
            ann = ann_by_id.get(sid)
            if ann and "fact_coverage" in ann:
                ann_judge_scores.append(r["overall"])
                ann_coverage.append(ann["fact_coverage"])

    n_ann = len(ann_judge_scores)
    if n_ann >= 2:
        rho_ann = stats_utils.spearman(ann_judge_scores, ann_coverage)
        # Pearson 简单实现
        import math
        mx, my = sum(ann_judge_scores) / n_ann, sum(ann_coverage) / n_ann
        cov = sum((ann_judge_scores[i] - mx) * (ann_coverage[i] - my) for i in range(n_ann)) / n_ann
        sx = math.sqrt(sum((x - mx) ** 2 for x in ann_judge_scores) / n_ann)
        sy = math.sqrt(sum((y - my) ** 2 for y in ann_coverage) / n_ann)
        pearson = cov / (sx * sy) if sx > 0 and sy > 0 else None
        mae_ann = sum(abs(ann_judge_scores[i] - ann_coverage[i] * 100) for i in range(n_ann)) / n_ann

        # 按 fact_coverage 分组看裁判分
        # 高 coverage (>0.3) vs 低 coverage (<=0.3)
        high_judge = [ann_judge_scores[i] for i in range(n_ann) if ann_coverage[i] > 0.3]
        low_judge = [ann_judge_scores[i] for i in range(n_ann) if ann_coverage[i] <= 0.3]
    else:
        rho_ann = pearson = mae_ann = None
        high_judge = low_judge = []

    ann_metrics = {
        "n": n_ann,
        "spearman_vs_fact_coverage": round(rho_ann, 3) if rho_ann is not None else None,
        "pearson_vs_fact_coverage": round(pearson, 3) if pearson is not None else None,
        "mae_vs_coverage_x100": round(mae_ann, 2) if mae_ann is not None else None,
        "high_coverage_n": len(high_judge),
        "high_coverage_mean_judge": round(sum(high_judge) / len(high_judge), 1) if high_judge else None,
        "low_coverage_n": len(low_judge),
        "low_coverage_mean_judge": round(sum(low_judge) / len(low_judge), 1) if low_judge else None,
    }

    # ========== 输出报告 ==========
    report = {
        "label_pool_28": label_metrics,
        "announcement_122": ann_metrics,
        "overall": {
            "n_total": sa["n_total"],
            "n_ok": sa["n_ok"],
            "parse_fail_ids": [r["id"] for r in sa["samples"] if r["status"] != "ok"],
        },
    }

    path = "src/results/strict_auditor_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("严格审计员全面打分分析")
    print("=" * 60)
    print(f"\n【28 条 label_pool vs 人工标注】n = {label_metrics.get('n', 0)}")
    for k, v in label_metrics.items():
        if k == "n":
            continue
        print(f"  {k:20s} = {v}")

    print(f"\n【122 条公告摘要 vs fact_coverage】n = {ann_metrics.get('n', 0)}")
    for k, v in ann_metrics.items():
        if k == "n":
            continue
        print(f"  {k:35s} = {v}")

    print(f"\n【整体】成功 {sa['n_ok']}/{sa['n_total']}，parse_fail: {report['overall']['parse_fail_ids']}")
    print(f"报告已写入 {path}")


if __name__ == "__main__":
    main()
