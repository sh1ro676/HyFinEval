"""Hy3-judge 交叉验证（C 组）。

对真实应用样本，分别用「规则 rubric」与「Hy3 多视角裁判」打分，
计算两者在真实模型输出上的 Spearman 秩相关，作为自动评分一致性的硬证据。

用法：
  python src/run_judge_cv.py                 # 默认分层抽样 24 条
  python src/run_judge_cv.py --n 40 --seed 7
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_store
import config
import hy3_app
import baseline_app
import evaluator


def stratified_sample(samples, n, seed=42):
    rng = __import__("random").Random(seed)
    by_diff = defaultdict(list)
    for s in samples:
        if s.get("subtask") == "评估器验证":
            continue
        by_diff[s.get("difficulty", "中")].append(s)
    # 尽量均衡覆盖各档
    out, per = [], max(1, n // max(1, len(by_diff)))
    for d in by_diff:
        rng.shuffle(by_diff[d])
        out.extend(by_diff[d][:per])
    rng.shuffle(out)
    return out[:n]


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for i, v in enumerate(order):
            r[v] = i + 1
        return r
    if len(a) < 2:
        return None
    ra, rb = rank(a), rank(b)
    m = sum(ra) / len(ra)
    cov = sum((ra[i] - m) * (rb[i] - m) for i in range(len(ra)))
    va = sum((x - m) ** 2 for x in ra) ** 0.5
    vb = sum((x - m) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-hy3-output", action="store_true",
                    help="用真实 Hy3 生成输出（默认用离线基线，确定性更强）")
    args = ap.parse_args()

    samples = stratified_sample(data_store.load_samples(), args.n, args.seed)
    rule, judge, ids = [], [], []
    for s in samples:
        out = (hy3_app.generate(s, use_rag=True) if args.use_hy3_output
               else baseline_app.baseline_generate(s))
        if not out:
            out = baseline_app.baseline_generate(s)
        ev = evaluator.evaluate(s, out, use_hy3_judge=True)
        if ev.get("hy3_judge_overall") is None:
            print(f"  {s['id']} judge 缺失，跳过")
            continue
        rule.append(ev["overall"])
        judge.append(ev["hy3_judge_overall"])
        ids.append(s["id"])
        print(f"  {s['id']} rule={ev['overall']:.1f} judge={ev['hy3_judge_overall']:.1f}")

    rho = spearman(rule, judge)
    mad = (sum(abs(rule[i] - judge[i]) for i in range(len(rule))) / len(rule)) if rule else None
    result = {
        "n": len(rule),
        "spearman_rule_vs_judge": round(rho, 3) if rho is not None else None,
        "mean_abs_diff": round(mad, 3) if mad is not None else None,
        "ids": ids,
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "judge_cv.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("=" * 50)
    print(f"交叉验证样本数 n = {len(rule)}")
    print(f"rubric vs Hy3-judge Spearman = {result['spearman_rule_vs_judge']}")
    print(f"平均绝对分差 = {result['mean_abs_diff']}")
    print(f"结果已写入 {config.RESULTS_DIR}/judge_cv.json")


if __name__ == "__main__":
    main()
