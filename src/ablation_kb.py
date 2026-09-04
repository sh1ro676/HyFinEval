# -*- coding: utf-8 -*-
"""最小验证：在 8 条 公告摘要 样本上做 KB-grounding 消融。

对照：
  baseline  = label_pool 中已存 auto_dims（factual 恒 0.9 / citation 恒 1.0）
  +KB       = factual/citation 改为 kb_grounding 查表核对，其余维度沿用原 auto_dims
分别计算与人工三档（优90/中60/差30）的 Spearman，验证「扩充 KB 能否解决失效」。

用法：python src/ablation_kb.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import stats_utils
import kb_grounding as kb

BAND_MAP = {"优": 90, "中": 60, "差": 30}


def main():
    pool = json.load(open(os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json"), encoding="utf-8"))
    lab = json.load(open(os.path.join(config.ROOT_DIR, "data_cache", "human_labels.json"), encoding="utf-8"))

    items = [p for p in pool if p.get("subtask") == "公告摘要"]
    rows = []
    base_auto, kb_auto, human = [], [], []
    for p in items:
        iid = p["id"]
        out = p.get("output") or ""
        inp = p.get("input") or ""
        dims = p.get("auto_dims") or {}
        if iid not in lab or "A" not in lab[iid]:
            continue
        band = lab[iid]["A"].get("band")
        if band not in BAND_MAP:
            continue
        meta = kb.match_announcement(inp)

        # baseline 维度（沿用存储值）
        d_base = dict(dims)
        # +KB 维度（仅替换 factual / citation）
        d_kb = dict(dims)
        if meta["matched"] and meta["true_type"]:
            d_kb["factual_accuracy"] = kb.grounded_factual(out, meta["true_type"])
        else:
            d_kb["factual_accuracy"] = 0.6
        d_kb["citation_verifiability"] = kb.grounded_citation(out)

        def _overall(d):
            return round(sum(d[k] * config.DIMENSION_WEIGHTS[k] for k in d) * 100, 1)

        base_s = _overall(d_base)
        kb_s = _overall(d_kb)
        h_s = BAND_MAP[band]

        base_auto.append(base_s)
        kb_auto.append(kb_s)
        human.append(h_s)
        rows.append({
            "id": iid, "band": band,
            "true_type": meta["true_type"], "matched": meta["matched"],
            "claimed_type": kb.extract_claimed_type(out),
            "factual_base": d_base.get("factual_accuracy"),
            "factual_kb": d_kb["factual_accuracy"],
            "citation_base": d_base.get("citation_verifiability"),
            "citation_kb": d_kb["citation_verifiability"],
            "auto_base": base_s, "auto_kb": kb_s, "human": h_s,
        })

    rho_base = stats_utils.spearman(base_auto, human)
    rho_kb = stats_utils.spearman(kb_auto, human)

    out = {
        "n": len(rows),
        "baseline_公告摘要_spearman": round(rho_base, 3) if rho_base is not None else None,
        "plusKB_公告摘要_spearman": round(rho_kb, 3) if rho_kb is not None else None,
        "delta": round(rho_kb - rho_base, 3) if (rho_base is not None and rho_kb is not None) else None,
        "rows": rows,
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "kb_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("公告摘要 KB-grounding 消融（n=%d）" % len(rows))
    print("  baseline  Spearman = %.3f" % (rho_base if rho_base is not None else float('nan')))
    print("  +KB       Spearman = %.3f" % (rho_kb if rho_kb is not None else float('nan')))
    print("  Δ                   = %.3f" % (rho_kb - rho_base if rho_kb is not None and rho_base is not None else float('nan')))
    print("-" * 60)
    for r in rows:
        print("  %-8s band=%-2s true=%-10s claimed=%-12s | fact %.1f->%.1f cit %.1f->%.1f | auto %.1f->%.1f (human %d)"
              % (r["id"], r["band"], r["true_type"], r["claimed_type"],
                 r["factual_base"], r["factual_kb"], r["citation_base"], r["citation_kb"],
                 r["auto_base"], r["auto_kb"], r["human"]))
    print("结果写入 %s" % os.path.join(config.RESULTS_DIR, "kb_ablation.json"))


if __name__ == "__main__":
    main()
