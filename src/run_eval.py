# -*- coding: utf-8 -*-
"""
运行完整评测：应用生成 -> 规则评估 -> 判别力/一致性/对抗性验证 -> 输出结果。
用法：
  python run_eval.py                  # 用基线应用 + 规则评估（无需 key，立即看效果）
  python run_eval.py --use-hy3       # 应用层启用 Hy3（需 HY3_API_KEY）
  python run_eval.py --use-hy3 --workers 10   # 接入推理模型时并发生成，显著提速
  python run_eval.py --use-hy3-judge # 应用用基线，仅裁判启用 Hy3 交叉验证
"""
import os
import sys
import json
import argparse
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import data_store
import baseline_app
import hy3_app
import evaluator
import config


def generate_output(sample, use_hy3_app):
    if use_hy3_app:
        out = hy3_app.generate(sample)
        if out:
            return out
    return baseline_app.baseline_generate(sample)


def spearman(a, b):
    """简易 Spearman：基于秩的相关（样本少时够用）。"""
    n = len(a)
    if n < 2:
        return None
    ra = _rank(a); rb = _rank(b)
    mean = sum(ra) / n
    cov = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    va = sum((x - mean) ** 2 for x in ra) ** 0.5
    vb = sum((x - mean) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    for r, i in enumerate(order):
        ranks[i] = r + 1
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hy3", action="store_true", help="应用层启用 Hy3")
    ap.add_argument("--use-hy3-judge", action="store_true", help="裁判启用 Hy3 交叉验证")
    ap.add_argument("--workers", type=int, default=1,
                    help="应用生成并发线程数（默认1串行；接入 Hy3 推理模型建议 8-10）")
    args = ap.parse_args()

    use_hy3_app = args.use_hy3
    use_judge = args.use_hy3_judge
    if use_hy3_app and not config.USE_HY3:
        print("[警告] 未检测到 HY3_API_KEY，回退到基线应用。")
        use_hy3_app = False

    samples = data_store.load_samples()
    app_samples = [s for s in samples if s["subtask"] != "评估器验证"]
    val_samples = [s for s in samples if s["subtask"] == "评估器验证"]

    # ---- 1. 应用评测 ----
    # 生成阶段为网络 IO 密集（尤其推理模型），可并发；评估阶段为本地规则，串行即可。
    app_results = []
    if use_hy3_app and args.workers > 1:
        print(f"[信息] 并发生成中（workers={args.workers}）...", flush=True)
        outs = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(generate_output, s, True): s["id"] for s in app_samples}
            for f in as_completed(futs):
                outs[futs[f]] = f.result()
        for s in app_samples:
            ev = evaluator.evaluate(s, outs[s["id"]], use_judge)
            app_results.append({"id": s["id"], "subtask": s["subtask"],
                                "difficulty": s["difficulty"], "counterfeit": s.get("is_counterfeit", False),
                                "human_rating": s.get("human_anchor_rating"), **ev})
    else:
        for s in app_samples:
            out = generate_output(s, use_hy3_app)
            ev = evaluator.evaluate(s, out, use_judge)
            app_results.append({"id": s["id"], "subtask": s["subtask"],
                                "difficulty": s["difficulty"], "counterfeit": s.get("is_counterfeit", False),
                                "human_rating": s.get("human_anchor_rating"), **ev})

    # ---- 2. 验证集判别力（输出=人工构造的好坏中差对抗样本）----
    val_results = []
    for s in val_samples:
        # 把 reference_output 当作"被评估的输出"
        out = {"answer": s["reference_output"], "citations": []}
        ev = evaluator.evaluate(s, out, use_judge)
        val_results.append({"id": s["id"], "difficulty": s["difficulty"],
                            "expected": s.get("human_anchor_rating"), **ev})

    # ---- 3. 统计 ----
    def avg(xs): return round(st.mean(xs), 1) if xs else 0
    app_overall = [r["overall"] for r in app_results]
    by_sub = {}
    for r in app_results:
        by_sub.setdefault(r["subtask"], []).append(r["overall"])
    by_diff = {}
    for r in app_results:
        by_diff.setdefault(r["difficulty"], []).append(r["overall"])

    # 判别力：好/中/差/对抗 平均分
    val_by_diff = {}
    for r in val_results:
        val_by_diff.setdefault(r["difficulty"], []).append(r["overall"])
    disc = {k: avg(v) for k, v in val_by_diff.items()}

    # 一致性（主）：验证集评估分 vs 人工锚定档位（好=A=100,中=B=80,差=D=40,对抗=D=40）
    val_rating_map = {"好": 100, "中": 80, "差": 40, "对抗": 40}
    val_pairs = [(r["overall"], val_rating_map[r["difficulty"]])
                 for r in val_results if r["difficulty"] in val_rating_map]
    corr = spearman([p[0] for p in val_pairs], [p[1] for p in val_pairs]) if val_pairs else None
    # 一致性（辅）：应用样本评估分 vs 人工锚定（基线版区分度参考，可能偏低）
    rating_map = {"A": 100, "B": 80, "C": 60, "D": 40}
    app_pairs = [(r["overall"], rating_map.get(r["human_rating"]))
                 for r in app_results if r["human_rating"] in rating_map]
    app_corr = spearman([p[0] for p in app_pairs], [p[1] for p in app_pairs]) if app_pairs else None

    # 对抗性：对抗样本平均分应低
    adv_avg = disc.get("对抗", None)

    summary = {
        "use_hy3_app": use_hy3_app,
        "use_hy3_judge": use_judge,
        "app_sample_count": len(app_results),
        "val_sample_count": len(val_results),
        "app_overall_avg": avg(app_overall),
        "by_subtask": {k: avg(v) for k, v in by_sub.items()},
        "by_difficulty": {k: avg(v) for k, v in by_diff.items()},
        "discrimination": disc,
        "consistency_spearman": round(corr, 3) if corr is not None else None,
        "consistency_app_spearman": round(app_corr, 3) if app_corr is not None else None,
        "adversarial_avg": adv_avg,
        "discrimination_ok": (disc.get("好", 0) > disc.get("中", 0) > disc.get("差", 0)
                              and disc.get("对抗", 100) < 50),
    }

    out_path = os.path.join(config.RESULTS_DIR, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "app_results": app_results,
                   "val_results": val_results}, f, ensure_ascii=False, indent=2)

    _print_report(summary, app_results, val_results)
    print(f"\n[完成] 结果已写入：{out_path}")


def _print_report(summary, app_results, val_results):
    print("=" * 60)
    print("A+B 路径评测报告（规则评估器）")
    print(f"应用层用 Hy3：{summary['use_hy3_app']} | 裁判用 Hy3：{summary['use_hy3_judge']}")
    print("=" * 60)
    print(f"应用样本数：{summary['app_sample_count']}  平均综合分：{summary['app_overall_avg']}")
    print("分任务平均分：", summary["by_subtask"])
    print("分难度平均分：", summary["by_difficulty"])
    print("-" * 60)
    print("判别力（验证集 好/中/差/对抗 平均分）：", summary["discrimination"])
    print(f"判别力达标：{summary['discrimination_ok']}")
    print(f"一致性 Spearman(验证集评估分 vs 人工锚定)：{summary['consistency_spearman']}")
    print(f"一致性 Spearman(应用样本评估分 vs 人工锚定)：{summary['consistency_app_spearman']}（基线版区分度参考）")
    print(f"对抗性（对抗样本平均分，应<50）：{summary['adversarial_avg']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
