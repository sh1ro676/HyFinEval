# -*- coding: utf-8 -*-
"""模拟成对比较锦标赛：用已有 committee 绝对分推导 pairwise 胜负 → BT 聚合 → 验证。

核心问题：成对比较形式是否能突破绝对分的 ceiling effect，提升与人类对齐度？
本脚本**不需要新 API 调用**，直接复用 committee_cv.json 的三角色分数。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bt_model
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


def compute_metrics(rule_scores, human_scores, human_bands):
    """计算 κ、Spearman、MAE、一致率。"""
    n = len(rule_scores)
    if n < 2:
        return None
    rho = stats_utils.spearman(rule_scores, human_scores)
    mae = sum(abs(rule_scores[i] - human_scores[i]) for i in range(n)) / n
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
    ap.add_argument("--out", default="src/results/pairwise_simulation.json")
    args = ap.parse_args()

    cv = load_json(args.committee)
    human = load_json(args.human)

    # 提取：ID -> 三角色分数 / 规则分 / 人工档
    samples = []
    for r in cv.get("samples", []):
        sid = r["id"]
        h = human.get(sid, {})
        ann = h.get("A") or h.get("B") or next(iter(h.values()), None) if isinstance(h, dict) else None
        if not ann:
            continue
        c = r.get("committee")
        if not c:
            continue
        role_scores = {}
        for role_res in c.get("role_results", []):
            role_scores[role_res["role"]] = float(role_res["overall"])
        if len(role_scores) < 3:
            continue
        samples.append({
            "id": sid,
            "rule_score": float(r["rule_score"]),
            "committee_mean": float(c["committee_mean"]),
            "role_scores": role_scores,
            "human_band": ann.get("band", "中"),
            "human_score": band_to_score(ann.get("band", "中")),
        })

    n = len(samples)
    ids = [s["id"] for s in samples]
    print(f"有效样本数: {n}")

    # ---- 基准：直接用绝对分 ----
    # 1) 规则分
    rule_scores_list = [s["rule_score"] for s in samples]
    human_scores_list = [s["human_score"] for s in samples]
    human_bands_list = [s["human_band"] for s in samples]
    rule_metrics = compute_metrics(rule_scores_list, human_scores_list, human_bands_list)

    # 2) 委员会平均分
    comm_scores_list = [s["committee_mean"] for s in samples]
    comm_metrics = compute_metrics(comm_scores_list, human_scores_list, human_bands_list)

    # ---- 模拟成对比较 + BT 聚合 ----
    bt_results = {}
    trans_results = {}

    for role_key in ["strict_auditor", "pragmatic_analyst", "casual_reader", "committee_mean"]:
        if role_key == "committee_mean":
            scores = {s["id"]: s["committee_mean"] for s in samples}
        else:
            scores = {s["id"]: s["role_scores"].get(role_key, 0) for s in samples}

        wins = bt_model.pairwise_from_scores(scores, ids)
        trans_rate = bt_model.pairwise_consistency(wins)
        trans_results[role_key] = round(trans_rate, 3)

        abilities = bt_model.bt_mle(wins, n, ids)
        if abilities:
            # BT 能力值先做 min-max 缩放再映射到 0-100，避免 softmax 压缩导致分布过窄
            raw_vals = [abilities[sid] for sid in ids]
            min_v, max_v = min(raw_vals), max(raw_vals)
            if max_v > min_v:
                bt_scores = [(v - min_v) / (max_v - min_v) * 100 for v in raw_vals]
            else:
                bt_scores = [50.0] * n
            bt_metrics = compute_metrics(bt_scores, human_scores_list, human_bands_list)
            bt_results[role_key] = bt_metrics
            bt_results[role_key]["transitivity_violation"] = round(trans_rate, 3)

    # ---- 三角色 BT 分取平均（委员会 BT 融合）----
    bt_fusion_scores = []
    for sid in ids:
        vals = []
        for role_key in ["strict_auditor", "pragmatic_analyst", "casual_reader"]:
            abilities = bt_model.bt_mle(
                bt_model.pairwise_from_scores(
                    {s["id"]: s["role_scores"].get(role_key, 0) for s in samples}, ids
                ), n, ids
            )
            if abilities:
                vals.append(abilities[sid])
        if vals:
            bt_fusion_scores.append(sum(vals) / len(vals))
        else:
            bt_fusion_scores.append(0)
    # min-max 缩放
    min_v, max_v = min(bt_fusion_scores), max(bt_fusion_scores)
    if max_v > min_v:
        bt_fusion_scores = [(v - min_v) / (max_v - min_v) * 100 for v in bt_fusion_scores]
    else:
        bt_fusion_scores = [50.0] * n

    bt_fusion_metrics = compute_metrics(bt_fusion_scores, human_scores_list, human_bands_list)

    # ---- 报告 ----
    report = {
        "n": n,
        "baseline": {
            "rule_score": rule_metrics,
            "committee_mean": comm_metrics,
        },
        "bt_per_role": bt_results,
        "bt_fusion": bt_fusion_metrics,
        "transitivity": trans_results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("模拟成对比较锦标赛报告（从绝对分推导 pairwise → BT 聚合）")
    print("=" * 70)
    print(f"\n【基准】n={n}")
    print(f"  规则分        : κ={rule_metrics['kappa']}  ρ={rule_metrics['spearman']}  MAE={rule_metrics['mae']}")
    print(f"  委员会平均分   : κ={comm_metrics['kappa']}  ρ={comm_metrics['spearman']}  MAE={comm_metrics['mae']}")
    print(f"\n【BT 聚合（模拟 pairwise）】")
    for role, m in bt_results.items():
        print(f"  {role:20s}: κ={m['kappa']}  ρ={m['spearman']}  MAE={m['mae']}  传递性违例={m['transitivity_violation']}")
    print(f"\n【BT 融合（三角色能力值平均）】")
    print(f"  κ={bt_fusion_metrics['kappa']}  ρ={bt_fusion_metrics['spearman']}  MAE={bt_fusion_metrics['mae']}")
    print(f"\n传递性违例率（越低越好）：")
    for role, tr in trans_results.items():
        print(f"  {role:20s}: {tr}")
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
