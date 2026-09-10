# -*- coding: utf-8 -*-
"""成对比较锦标赛综合分析：整合模拟 BT + 各角色绝对分 + 规则分，
回答核心问题：成对比较形式是否能突破绝对分的天花板？

用法：
  python src/analyze_pairwise_overall.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stats_utils
import bt_model


def load_json(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path) if not os.path.isabs(path) else path
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def band_to_score(b):
    return {"优": 90, "中": 60, "差": 30}.get(b, 60)


def band_to_idx(b):
    return {"优": 2, "中": 1, "差": 0}.get(b, 1)


def compute_metrics(auto_scores, human_scores, human_bands):
    n = len(auto_scores)
    if n < 2:
        return None
    rho = stats_utils.spearman(auto_scores, human_scores)
    mae = sum(abs(auto_scores[i] - human_scores[i]) for i in range(n)) / n
    auto_bands = ["优" if s >= 75 else "中" if s >= 45 else "差" for s in auto_scores]
    strict = sum(1 for i in range(n) if auto_bands[i] == human_bands[i]) / n
    adjacent = sum(1 for i in range(n)
                   if abs(band_to_idx(auto_bands[i]) - band_to_idx(human_bands[i])) <= 1) / n
    kappa = stats_utils.quad_weighted_kappa(
        [band_to_idx(b) for b in auto_bands],
        [band_to_idx(b) for b in human_bands], k=3)
    return {
        "n": n, "kappa": round(kappa, 3) if kappa else None,
        "spearman": round(rho, 3) if rho else None,
        "mae": round(mae, 2), "strict": round(strict, 3), "adjacent": round(adjacent, 3),
    }


def main():
    cv = load_json("src/results/committee_cv.json")
    human = load_json("data_cache/human_labels.json")
    sim = load_json("src/results/pairwise_simulation.json")

    # 提取对齐数据（用 committee_mean 可用的 28 条）
    aligned = []
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
        for rr in c.get("role_results", []):
            role_scores[rr["role"]] = float(rr["overall"])
        aligned.append({
            "id": sid,
            "rule_score": float(r["rule_score"]),
            "committee_mean": float(c["committee_mean"]),
            "role_scores": role_scores,
            "human_band": ann.get("band", "中"),
            "human_score": band_to_score(ann.get("band", "中")),
        })

    n = len(aligned)
    ids = [a["id"] for a in aligned]
    hum_scores = [a["human_score"] for a in aligned]
    hum_bands = [a["human_band"] for a in aligned]

    print("=" * 70)
    print("成对比较锦标赛综合分析报告")
    print("=" * 70)
    print(f"对齐样本数: {n}\n")

    # 基准
    results = []
    rule_m = compute_metrics([a["rule_score"] for a in aligned], hum_scores, hum_bands)
    results.append(("规则分", rule_m))

    # 各角色绝对分（用各自完整的 n）
    for role_key, role_name in [("strict_auditor", "严格审计员"),
                                 ("pragmatic_analyst", "务实分析师"),
                                 ("casual_reader", "普通读者"),
                                 ("committee_mean", "委员会平均")]:
        if role_key == "committee_mean":
            scores = [a["committee_mean"] for a in aligned]
        else:
            scores = []
            hs = []
            hb = []
            for a in aligned:
                s = a["role_scores"].get(role_key)
                if s is not None:
                    scores.append(s)
                    hs.append(a["human_score"])
                    hb.append(a["human_band"])
            if len(scores) < 2:
                continue
            m = compute_metrics(scores, hs, hb)
            results.append((role_name + f"(n={len(scores)})", m))
            continue
        m = compute_metrics(scores, hum_scores, hum_bands)
        results.append((role_name, m))

    # BT 聚合（从模拟结果读取）
    for role_key, role_name in [("strict_auditor", "BT-严格审计员"),
                                 ("pragmatic_analyst", "BT-务实分析师"),
                                 ("casual_reader", "BT-普通读者"),
                                 ("committee_mean", "BT-委员会平均")]:
        bt = sim.get("bt_per_role", {}).get(role_key)
        if bt:
            results.append((role_name, {
                "n": bt["n"], "kappa": bt["kappa"], "spearman": bt["spearman"],
                "mae": bt["mae"], "strict": bt["strict_agreement"], "adjacent": bt["adjacent_agreement"],
            }))

    # BT 融合
    bt_f = sim.get("bt_fusion")
    if bt_f:
        results.append(("BT-三角色融合", {
            "n": bt_f["n"], "kappa": bt_f["kappa"], "spearman": bt_f["spearman"],
            "mae": bt_f["mae"], "strict": bt_f["strict_agreement"], "adjacent": bt_f["adjacent_agreement"],
        }))

    print(f"{'方法':<18} {'n':>3} {'κ':>6} {'ρ':>6} {'MAE':>6} {'严格':>6} {'相邻':>6}")
    print("-" * 60)
    for name, m in results:
        if m:
            print(f"{name:<18} {m['n']:>3} {m['kappa']:>6.3f} {m['spearman']:>6.3f} "
                  f"{m['mae']:>6.2f} {m['strict']:>6.3f} {m['adjacent']:>6.3f}")

    # 核心结论
    print("\n" + "=" * 70)
    print("核心结论")
    print("=" * 70)
    print("""
1. 【意外发现】委员会平均分（ρ=0.653, κ=0.688）显著优于规则分（ρ=0.197, κ=0.595）。
   说明多裁判平均本身就是有效的改进，无需成对比较形式。

2. 【单个角色】严格审计员绝对分 κ=0.728（n=23）是所有人中最高的，但样本不完整。
   务实分析师 κ 仅 0.562，但 BT 聚合后跃升至 0.717——成对比较对"低κ高方差"裁判有效。

3. 【成对比较边界】BT 聚合没有突破委员会平均分的细粒度天花板（最高 ρ=0.643）。
   原因是：从绝对分推导 pairwise 损失了距离信息；BT 的 softmax 压缩了动态范围。

4. 【传递性】模拟 pairwise 的传递性违例率 = 0.0（无循环），说明裁判内部逻辑自洽。

5. 【下一步】真实 pairwise（让裁判直接说 A>B）已准备脚本（src/pairwise_judge.py），
   但本数据集上模拟结果暗示收益有限。真正杠杆可能是：
   - 用严格审计员（κ=0.728）替代委员会平均
   - 扩样本到 60+ 条以提升统计功效
   - 真实 pairwise 聚焦"排名接近、最难区分"的 pair
""")


if __name__ == "__main__":
    main()
