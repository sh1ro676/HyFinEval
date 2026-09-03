"""离线重算已存评测结果（eval_results*.json）的 overall / dimensions。

不重新生成模型输出（保留原始 output 文本），仅用【修复后的 evaluator】
重新打分，从而得到与新 rubric 逻辑一致的总分与维度分。

用法：
  python src/rescore_existing.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_store
import config
import evaluator


def _rescore_file(path):
    d = json.load(open(path, encoding="utf-8"))
    samples = {s["id"]: s for s in data_store.load_samples()}
    for key in ("app_results", "val_results"):
        rows = d.get(key) or []
        for r in rows:
            sid = r["id"]
            s = samples.get(sid)
            if not s:
                continue
            out = {"answer": r.get("output") or ""}
            ev = evaluator.evaluate(s, out)
            r["dimensions"] = {k: round(float(v), 3) for k, v in ev["dimensions"].items()}
            r["overall"] = ev["overall"]
            r["failure_mode"] = ev["failure_mode"]
            r["compliance_breaker"] = ev["compliance_breaker"]
    # 重新汇总 summary
    _resummarize(d)
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  已重算并写回 {path}")
    return d


def _resummarize(d):
    app = d.get("app_results") or []
    val = d.get("val_results") or []
    w = config.DIMENSION_WEIGHTS

    def avg_overall(rows):
        return round(sum(r["overall"] for r in rows) / len(rows), 1) if rows else None

    def by_group(rows, field):
        g = defaultdict(list)
        for r in rows:
            g[r.get(field, "未知")].append(r["overall"])
        return {k: round(sum(v) / len(v), 1) for k, v in g.items()}

    def dim_means(rows):
        means = {}
        for dim in w:
            vals = [r["dimensions"].get(dim) for r in rows if r.get("dimensions")]
            means[dim] = round(sum(vals) / len(vals), 3) if vals else None
        return means

    summary = d.get("summary", {})
    summary["app_overall_avg"] = avg_overall(app)
    summary["by_subtask"] = by_group(app, "subtask")
    summary["by_difficulty"] = by_group(app, "difficulty")
    # 维度均值（开闭卷对照用）
    summary["dim_means_app"] = dim_means(app)
    if val:
        # 验证集判别力（好/中/差/对抗）
        disc = defaultdict(list)
        for r in val:
            disc[r.get("difficulty")].append(r["overall"])
        summary["discrimination"] = {k: round(sum(v) / len(v), 1)
                                    for k, v in disc.items()}
        # 内部一致性：overall vs 构造档位（好100/中80/差40/对抗40）
        anchor = {"好": 100, "中": 80, "差": 40, "对抗": 40}
        a = [r["overall"] for r in val]
        b = [anchor.get(r.get("difficulty")) for r in val]
        try:
            from stats_utils import spearman
            summary["consistency_spearman"] = round(spearman(a, b), 3)
        except Exception:
            pass
        summary["adversarial_avg"] = summary["discrimination"].get("对抗")
    if app:
        # 应用集 A/D 二值（设计不当，仅保留数值供参考）
        anchor2 = {"A": 100, "D": 40}
        a2 = [r["overall"] for r in app if r.get("human_rating") in anchor2]
        b2 = [anchor2[r["human_rating"]] for r in app if r.get("human_rating") in anchor2]
        try:
            from stats_utils import spearman
            summary["consistency_app_spearman"] = round(spearman(a2, b2), 3)
        except Exception:
            pass
    d["summary"] = summary


if __name__ == "__main__":
    base = os.path.join(config.RESULTS_DIR)
    for fn in ["eval_results.json", "eval_results_norag.json"]:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            print(f"处理 {fn} ...")
            _rescore_file(p)
