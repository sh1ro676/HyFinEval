# -*- coding: utf-8 -*-
"""
稳定性 / 重测一致性脚本：同一批样本重复生成+评估 N 次，统计评估分数的方差与重测一致性。
用法：
  python run_stability.py --runs 3                 # 基线（确定性，应方差≈0，证评估器可复现）
  python run_stability.py --use-hy3 --runs 3       # 接入 Hy3，测真实 LLM 输出方差（需 HY3_API_KEY）
  python run_stability.py --use-hy3 --no-rag --runs 3
输出：每条样本 overall 的 mean/std；全样本 test-retest Spearman（run1 vs runN）。
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
import stats_utils


def generate_output(sample, use_hy3_app, use_rag=True):
    if use_hy3_app:
        out = hy3_app.generate(sample, use_rag)
        if out:
            return out
    return baseline_app.baseline_generate(sample)


def spearman(a, b):
    """委托到 stats_utils：带并列修正的平均秩实现（已与 scipy 对齐）。"""
    return stats_utils.spearman(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-hy3", action="store_true", help="应用层启用 Hy3（测真实 LLM 方差）")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    use_hy3_app = args.use_hy3
    use_rag = not args.no_rag
    if use_hy3_app and not config.USE_HY3:
        print("[警告] 未检测到 HY3_API_KEY，回退基线。")
        use_hy3_app = False

    samples = [s for s in data_store.load_samples() if s["subtask"] != "评估器验证"]
    per_id = {s["id"]: [] for s in samples}

    for run in range(args.runs):
        print(f"[信息] 第 {run + 1}/{args.runs} 轮生成+评估中...", flush=True)
        outs = {}
        if use_hy3_app and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(generate_output, s, True, use_rag): s["id"] for s in samples}
                for f in as_completed(futs):
                    outs[futs[f]] = f.result()
        else:
            for s in samples:
                outs[s["id"]] = generate_output(s, use_hy3_app, use_rag)
        for s in samples:
            ev = evaluator.evaluate(s, outs[s["id"]])
            per_id[s["id"]].append(ev["overall"])

    # 统计
    stds = []
    for sid, scores in per_id.items():
        if len(scores) > 1:
            stds.append(st.pstdev(scores))
    mean_std = round(st.mean(stds), 2) if stds else 0.0
    max_std = round(max(stds), 2) if stds else 0.0

    # test-retest Spearman: run0 vs last run（整体排序稳定性）
    r1 = [per_id[s["id"]][0] for s in samples]
    rN = [per_id[s["id"]][-1] for s in samples]
    corr = spearman(r1, rN)

    summary = {
        "runs": args.runs,
        "use_hy3_app": use_hy3_app,
        "use_rag": use_rag,
        "sample_count": len(samples),
        "mean_std_overall": mean_std,
        "max_std_overall": max_std,
        "test_retest_spearman_run1_vs_last": round(corr, 3) if corr is not None else None,
        "interpretation": ("基线确定性：方差≈0 证明评估器可复现；"
                           "Hy3 模式：方差反映真实 LLM 输出波动，越小越稳健。"),
    }
    print("=" * 60)
    print("稳定性 / 重测一致性报告")
    print("=" * 60)
    print(f"轮数：{summary['runs']} | 样本数：{summary['sample_count']} | 用Hy3：{use_hy3_app}")
    print(f"overall 平均标准差：{mean_std}（最大 {max_std}）")
    print(f"重测一致性 Spearman(run1 vs 末轮)：{summary['test_retest_spearman_run1_vs_last']}")
    print("=" * 60)

    out_path = os.path.join(config.RESULTS_DIR, "stability_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_id": per_id}, f, ensure_ascii=False, indent=2)
    print(f"[完成] 结果已写入：{out_path}")


if __name__ == "__main__":
    main()
