# -*- coding: utf-8 -*-
"""真实成对比较：让 Hy3 裁判直接比较 "A vs B 哪个更好"。

采样策略（高效）：按 committee_mean 排序后，每个样本只与前后 2 个邻居比较，
聚焦"排名接近、最难区分"的 pair。每对做左右互换消除位置偏置。

用法：
  python src/pairwise_judge.py --workers 10
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import hy3_app


def load_json(path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path) if not os.path.isabs(path) else path
    with open(full, encoding="utf-8") as f:
        return json.load(f)


PAIRWISE_PROMPT = (
    "你是一位金融分析质量评审员。请将以下两份模型输出进行直接比较，"
    "判断哪一份质量更高。评估维度：事实准确性、引用可靠性、逻辑清晰度、"
    "有无胡编乱造、对不确定信息的处理是否恰当。\n\n"
    "只回答一个字：A 或 B。不要解释理由。"
)


def compare_pair(sample_a, output_a, sample_b, output_b, swap: bool = False):
    """调用 Hy3 比较两个输出。swap=True 时互换 A/B 位置以检测位置偏置。"""
    if not config.USE_HY3:
        return None
    if swap:
        sample_a, sample_b = sample_b, sample_a
        output_a, output_b = output_b, output_a

    text_a = output_a.get("answer", "") if isinstance(output_a, dict) else str(output_a)
    text_b = output_b.get("answer", "") if isinstance(output_b, dict) else str(output_b)

    user = (
        f"【题目】{sample_a.get('input', '')}\n\n"
        f"--- 输出 A ---\n{text_a[:1500]}\n\n"
        f"--- 输出 B ---\n{text_b[:1500]}\n\n"
        f"哪一份输出质量更高？只回答 A 或 B。"
    )
    res = hy3_app.call_hy3([
        {"role": "system", "content": PAIRWISE_PROMPT},
        {"role": "user", "content": user},
    ], temperature=0.0, max_tokens=10)
    if not res:
        return None
    res = res.strip().upper()
    if "A" in res and "B" not in res:
        winner = "B" if swap else "A"
    elif "B" in res and "A" not in res:
        winner = "A" if swap else "B"
    else:
        return None
    return {"winner": winner, "swap": swap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committee", default="src/results/committee_cv.json")
    ap.add_argument("--out", default="src/results/pairwise_judge.json")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--neighbor", type=int, default=2, help="每个样本与前后几个邻居比较")
    args = ap.parse_args()

    if not config.USE_HY3:
        print("错误：未检测到 HY3_API_KEY，无法进行成对比较。")
        return

    cv = load_json(args.committee)

    # 提取有效样本（有 committee_mean 的）
    samples = []
    for r in cv.get("samples", []):
        c = r.get("committee")
        if not c:
            continue
        samples.append({
            "id": r["id"],
            "input": r.get("input", ""),
            "output": {"answer": r.get("output", ""), "citations": r.get("citations", [])},
            "committee_mean": c.get("committee_mean", 0),
        })

    n = len(samples)
    if n < 3:
        print(f"有效样本仅 {n} 条，不足以做 pairwise。")
        return

    # 按 committee_mean 排序
    samples.sort(key=lambda x: x["committee_mean"])
    id_to_idx = {s["id"]: i for i, s in enumerate(samples)}

    # 生成邻居 pair
    pairs = []
    for i in range(n):
        for offset in range(1, args.neighbor + 1):
            j = i + offset
            if j >= n:
                continue
            a, b = samples[i], samples[j]
            # 避免重复 pair
            if (a["id"], b["id"]) not in [(p["a"], p["b"]) for p in pairs]:
                pairs.append({"a": a["id"], "b": b["id"], "sample_a": a, "sample_b": b})

    print(f"样本数: {n} | 生成 pair 数: {len(pairs)} | 含左右互换共 {len(pairs)*2} 次调用")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for p in pairs:
            # 原始顺序
            futures[ex.submit(compare_pair, p["sample_a"], p["sample_a"]["output"],
                              p["sample_b"], p["sample_b"]["output"], False)] = (p, False)
            # 互换顺序
            futures[ex.submit(compare_pair, p["sample_a"], p["sample_a"]["output"],
                              p["sample_b"], p["sample_b"]["output"], True)] = (p, True)

        for fut in as_completed(futures):
            p, swapped = futures[fut]
            try:
                r = fut.result()
                if r:
                    results.append({
                        "a": p["a"],
                        "b": p["b"],
                        "swap": swapped,
                        "winner": r["winner"],
                    })
            except Exception as e:
                print(f"  {p['a']} vs {p['b']} (swap={swapped}) 失败: {e}")

    # 统计位置偏置：互换前后结果是否一致
    consistent = 0
    total_swap_pairs = 0
    pair_results = {}
    for r in results:
        key = tuple(sorted([r["a"], r["b"]]))
        if key not in pair_results:
            pair_results[key] = {"normal": None, "swap": None}
        if r["swap"]:
            pair_results[key]["swap"] = r["winner"]
        else:
            pair_results[key]["normal"] = r["winner"]

    for key, vals in pair_results.items():
        if vals["normal"] is not None and vals["swap"] is not None:
            total_swap_pairs += 1
            # 互换后 winner 应该翻转：正常顺序 winner=A 意味着 A>B；swap 后 winner=B 也意味着 A>B
            if (vals["normal"] == key[0] and vals["swap"] == key[0]) or \
               (vals["normal"] == key[1] and vals["swap"] == key[1]):
                consistent += 1

    swap_consistency = consistent / total_swap_pairs if total_swap_pairs > 0 else 0

    report = {
        "n_samples": n,
        "n_pairs": len(pairs),
        "n_comparisons": len(results),
        "swap_consistency": round(swap_consistency, 3),
        "comparisons": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"成对比较完成：{len(results)}/{len(pairs)*2} 次调用成功")
    print(f"左右互换一致性: {swap_consistency:.1%} ({consistent}/{total_swap_pairs})")
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
