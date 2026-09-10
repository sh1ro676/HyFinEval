#!/usr/bin/env python3
"""从严格审计员绝对分推导 pairwise 模拟验证。

真实 pairwise 因 API 限流（单条 30s 且频繁返回 None）在当前条件下无法可靠跑通。
本脚本用已有的严格审计员绝对分做推导 pairwise → BT 聚合 → 与人工标注对齐，
作为真实 pairwise 的替代验证。

用法：
  python src/simulate_pairwise_strict.py
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


def pairwise_from_scores(scores, ids):
    """从绝对分推导 pairwise 胜负矩阵。"""
    wins = {}
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i == j:
                continue
            if scores[a] > scores[b]:
                wins[(a, b)] = wins.get((a, b), 0) + 1
    return wins


def bt_mle(wins, n, ids, max_iter=100, tol=1e-5):
    import statistics
    idx_map = {sid: i for i, sid in enumerate(ids)}
    abilities = [0.0] * n
    for _ in range(max_iter):
        old = abilities[:]
        for i in range(n):
            sid_i = ids[i]
            numer = 0.0
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                sid_j = ids[j]
                w_ij = wins.get((sid_i, sid_j), 0)
                w_ji = wins.get((sid_j, sid_i), 0)
                total = w_ij + w_ji
                if total == 0:
                    continue
                numer += total * (w_ij / total)
                pi = 1.0 / (1.0 + (abilities[j] - abilities[i]) + 1e-10)
                denom += total * pi
            if denom > 0:
                abilities[i] = numer / denom
        if all(abs(abilities[i] - old[i]) < tol for i in range(n)):
            break
    mu = statistics.mean(abilities)
    sd = statistics.stdev(abilities) if n > 1 else 1.0
    if sd == 0:
        sd = 1.0
    return {sid: (abilities[idx_map[sid]] - mu) / sd for sid in ids}


def main():
    # ---- 加载数据 ----
    with open("src/results/strict_auditor_all.json", encoding="utf-8") as f:
        sa = json.load(f)
    with open("data_cache/human_labels.json", encoding="utf-8") as f:
        human = json.load(f)

    # 只取 label_pool 的 28 条
    judge_scores = {}
    for r in sa.get("samples", []):
        if r.get("status") == "ok" and r["id"].startswith("FIN-") and r["id"] in human:
            judge_scores[r["id"]] = r["overall"]

    ids = sorted(judge_scores.keys())
    n = len(ids)
    print(f"严格审计员有效样本: {n} 条")

    if n < 3:
        print("样本不足，退出")
        return

    # ---- pairwise 推导 ----
    wins = pairwise_from_scores(judge_scores, ids)
    abilities = bt_mle(wins, n, ids)

    # ---- min-max 缩放到 0-100 ----
    vals = [abilities[sid] for sid in ids]
    min_v, max_v = min(vals), max(vals)
    range_v = max_v - min_v if max_v > min_v else 1.0
    bt_scores = {sid: (abilities[sid] - min_v) / range_v * 100 for sid in ids}

    # ---- 与人工标注对齐 ----
    auto_scores = []
    human_scores = []
    human_bands = []
    for sid in ids:
        h = human.get(sid, {})
        ann = h.get("A") or h.get("B") or next(iter(h.values()), None) if isinstance(h, dict) else None
        if not ann:
            continue
        auto_scores.append(bt_scores[sid])
        human_scores.append(band_to_score(ann.get("band", "中")))
        human_bands.append(ann.get("band", "中"))

    n_align = len(auto_scores)
    rho = stats_utils.spearman(auto_scores, human_scores)
    mae = sum(abs(auto_scores[i] - human_scores[i]) for i in range(n_align)) / n_align
    auto_bands = ["优" if s >= 75 else "中" if s >= 45 else "差" for s in auto_scores]
    kappa = stats_utils.quad_weighted_kappa(
        [band_to_idx(b) for b in auto_bands],
        [band_to_idx(b) for b in human_bands],
        k=3,
    )

    # ---- 与绝对分对比 ----
    abs_scores = [judge_scores[sid] for sid in ids]
    abs_bands = ["优" if s >= 75 else "中" if s >= 45 else "差" for s in abs_scores]
    rho_abs = stats_utils.spearman(abs_scores, human_scores)
    mae_abs = sum(abs(abs_scores[i] - human_scores[i]) for i in range(n_align)) / n_align
    kappa_abs = stats_utils.quad_weighted_kappa(
        [band_to_idx(b) for b in abs_bands],
        [band_to_idx(b) for b in human_bands],
        k=3,
    )

    report = {
        "n": n_align,
        "strict_auditor_bt": {
            "kappa": round(kappa, 3),
            "spearman": round(rho, 3),
            "mae": round(mae, 2),
        },
        "strict_auditor_absolute": {
            "kappa": round(kappa_abs, 3),
            "spearman": round(rho_abs, 3),
            "mae": round(mae_abs, 2),
        },
        "bt_scores": bt_scores,
        "absolute_scores": {sid: judge_scores[sid] for sid in ids},
    }

    out_path = "src/results/pairwise_strict_simulation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("严格审计员 pairwise 模拟验证（从绝对分推导）")
    print("=" * 60)
    print(f"\n对齐样本数: {n_align}")
    print(f"\n| 方法 | κ | ρ | MAE |")
    print(f"|---|---|---|---|")
    print(f"| BT 聚合 | {kappa:.3f} | {rho:.3f} | {mae:.2f} |")
    print(f"| 严格审计员绝对分 | {kappa_abs:.3f} | {rho_abs:.3f} | {mae_abs:.2f} |")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
