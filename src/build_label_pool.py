"""生成人工标注池（B 组前置）。

对分层抽样后的应用样本，用真实 Hy3（开卷）生成输出，连同输入、构造难度、规则自动分
一并存入 data_cache/label_pool.json，供 human_label.py 盲标与 label_stats.py 统计。

用法：
  python src/build_label_pool.py            # 默认 28 条
  python src/build_label_pool.py --n 30 --seed 7
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

CACHE = os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json")


def stratified(samples, n, seed=42):
    rng = __import__("random").Random(seed)
    by_diff = defaultdict(list)
    for s in samples:
        if s.get("subtask") == "评估器验证":
            continue
        by_diff[s.get("difficulty", "中")].append(s)
    out, per = [], max(1, n // max(1, len(by_diff)))
    for d in by_diff:
        rng.shuffle(by_diff[d])
        out.extend(by_diff[d][:per])
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not config.USE_HY3:
        print("⚠️ 未检测到 HY3_API_KEY，将用离线基线生成标注池（确定性，但输出同质）。")

    samples = stratified(data_store.load_samples(), args.n, args.seed)
    pool = []
    for s in samples:
        out = hy3_app.generate(s, use_rag=True) if config.USE_HY3 else None
        if not out:
            out = baseline_app.baseline_generate(s)
        ev = evaluator.evaluate(s, out)
        pool.append({
            "id": s["id"],
            "subtask": s.get("subtask"),
            "difficulty": s.get("difficulty"),
            "input": s.get("input"),
            "output": out.get("answer") if isinstance(out, dict) else str(out),
            "citations": out.get("citations") if isinstance(out, dict) else [],
            "auto_score": round(ev["overall"], 1),
        })
        print(f"  {s['id']} auto={ev['overall']:.1f}")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"标注池已写入 {CACHE}（{len(pool)} 条）")


if __name__ == "__main__":
    main()
