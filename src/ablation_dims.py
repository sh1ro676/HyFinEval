"""方案 A 消融：在 28 条真人盲标上对比「原 rubric 三维」与「变体三维」的排序对齐。

输出：
  - 总体 ρ（auto vs 人工档位）：原 vs 变体
  - 各子任务 ρ：原 vs 变体
  - 三个目标维度（format/citation/calibration）的：取值数、分布、与人工的 Spearman
    （原 vs 变体，验证方差是否释放、方向是否转正）
结果写入 src/results/ablation_dims.json。
"""

import json
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import evaluator
import stats_utils as su
from score_variants import variant_rule_evaluate

POOL = os.path.join("..", "data_cache", "label_pool.json")
LAB = os.path.join("..", "data_cache", "human_labels.json")
OUT = os.path.join("..", "src", "results", "ablation_dims.json")

BM = {"优": 90, "中": 60, "差": 30}
TARGET = ["format", "citation_verifiability", "calibration"]


def band_num(b):
    return BM.get(b, 60)


def main():
    pool = json.load(open(POOL, encoding="utf-8"))
    lab = json.load(open(LAB, encoding="utf-8"))
    rows = []
    for p in pool:
        if p["id"] not in lab:
            continue
        s = {"subtask": p["subtask"], "input": p["input"], "difficulty": p.get("difficulty", "中")}
        o = {"answer": p.get("output", ""), "citations": p.get("citations", [])}
        d_o, ov_o, _ = evaluator._rule_evaluate(s, o)
        d_v, ov_v, _ = variant_rule_evaluate(s, o)
        h = band_num(lab[p["id"]]["A"]["band"])
        rows.append({
            "id": p["id"], "subtask": p["subtask"],
            "orig_overall": ov_o, "var_overall": ov_v, "human": h,
            "orig_dims": d_o, "var_dims": d_v,
        })

    def rho_of(key):
        a = [r[key] for r in rows]
        h = [r["human"] for r in rows]
        return su.spearman(a, h)

    res = {}
    res["n"] = len(rows)
    res["overall_rho"] = {"orig": round(rho_of("orig_overall"), 3),
                          "variant": round(rho_of("var_overall"), 3)}

    # 各子任务
    by = collections.defaultdict(list)
    for r in rows:
        by[r["subtask"]].append(r)
    res["per_subtask"] = {}
    for st, rs in sorted(by.items()):
        a_o = [r["orig_overall"] for r in rs]
        a_v = [r["var_overall"] for r in rs]
        hh = [r["human"] for r in rs]
        res["per_subtask"][st] = {
            "n": len(rs),
            "rho_orig": round(su.spearman(a_o, hh), 3),
            "rho_variant": round(su.spearman(a_v, hh), 3),
        }

    # 逐维度诊断
    res["per_dimension"] = {}
    for dim in TARGET:
        od = [r["orig_dims"].get(dim) for r in rows]
        vd = [r["var_dims"].get(dim) for r in rows]
        hh = [r["human"] for r in rows]
        res["per_dimension"][dim] = {
            "orig": {
                "n_unique": len(set(od)),
                "mode_share": round(max(collections.Counter(od).values()) / len(od), 3),
                "spearman_vs_human": (round(su.spearman(od, hh), 3)
                                      if len(set(od)) > 1 else None),
            },
            "variant": {
                "n_unique": len(set(vd)),
                "mode_share": round(max(collections.Counter(vd).values()) / len(vd), 3),
                "spearman_vs_human": (round(su.spearman(vd, hh), 3)
                                      if len(set(vd)) > 1 else None),
                "min": round(min(vd), 3), "max": round(max(vd), 3),
                "mean": round(sum(vd) / len(vd), 3),
            },
        }
        # 分布（变体，取值≤6 时列出）
        c = collections.Counter(round(v, 3) for v in vd)
        if len(c) <= 8:
            res["per_dimension"][dim]["variant"]["dist"] = {str(k): v for k, v in sorted(c.items())}

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 打印
    print("方案 A 消融结果（n=%d）" % len(rows))
    print("  总体 ρ：  原 %+.3f  →  变体 %+.3f   (Δ %+.3f)" % (
        res["overall_rho"]["orig"], res["overall_rho"]["variant"],
        res["overall_rho"]["variant"] - res["overall_rho"]["orig"]))
    print("  各子任务 ρ（原 → 变体）：")
    for st, d in res["per_subtask"].items():
        print("    %-8s n=%2d  %+.3f → %+.3f" % (st, d["n"], d["rho_orig"], d["rho_variant"]))
    print("  目标维度诊断：")
    for dim in TARGET:
        o = res["per_dimension"][dim]["orig"]
        v = res["per_dimension"][dim]["variant"]
        print("    %-22s 原[u=%d,mode=%.2f,ρ=%s] → 变体[u=%d,mode=%.2f,ρ=%s,范围%.2f~%.2f]" % (
            dim, o["n_unique"], o["mode_share"], o["spearman_vs_human"],
            v["n_unique"], v["mode_share"], v["spearman_vs_human"], v["min"], v["max"]))


if __name__ == "__main__":
    main()
