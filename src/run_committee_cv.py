# -*- coding: utf-8 -*-
"""裁判委员会交叉验证（C2 组）。

对 label_pool 中已有模型输出，分别用「规则 rubric」与「三角色裁判委员会」打分，
输出平均分与分歧度（标准差）。分歧度本身作为「自动评分不确定性」的量化信号。

用法：
  python src/run_committee_cv.py              # 默认对 label_pool 全部 28 条
  python src/run_committee_cv.py --out path   # 指定输出路径
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import evaluator
import hy3_app
import stats_utils


def load_label_pool(path="data_cache/label_pool.json"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = os.path.join(root, path)
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def band_to_score(band: str) -> int:
    return {"优": 90, "中": 60, "差": 30}.get(band, 60)


def process_one(item: dict, use_hy3: bool) -> dict:
    sample = {
        "id": item["id"],
        "subtask": item.get("subtask", ""),
        "difficulty": item.get("difficulty", "中"),
        "input": item.get("input", ""),
        "reference_output": item.get("reference_output", ""),
    }
    output = {
        "answer": item.get("output", ""),
        "citations": item.get("citations", []),
    }
    # 规则分（确定性，离线可算）
    rule_ev = evaluator.evaluate(sample, output, use_hy3_judge=False, use_committee=False)
    result = {
        "id": item["id"],
        "rule_score": rule_ev["overall"],
        "rule_dims": rule_ev["dimensions"],
    }
    # 委员会分（需 Hy3 key）
    if use_hy3:
        c = evaluator.hy3_committee(sample, output)
        if c:
            result["committee"] = c
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "committee_cv.json"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    pool = load_label_pool()
    use_hy3 = config.USE_HY3
    if not use_hy3:
        print("警告：未检测到 HY3_API_KEY，委员会打分将不可用（仅输出规则分）。")

    results = []
    if use_hy3:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, item, True): item for item in pool}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    cid = r.get("committee", {})
                    print(f"  {r['id']} rule={r['rule_score']:.1f} "
                          f"committee_mean={cid.get('committee_mean','N/A')} "
                          f"std={cid.get('committee_std','N/A')} "
                          f"reliability={cid.get('committee_reliability','N/A')}")
                except Exception as e:
                    print(f"  {item['id']} 失败: {e}")
    else:
        for item in pool:
            r = process_one(item, False)
            results.append(r)
            print(f"  {r['id']} rule={r['rule_score']:.1f} (无 key，跳过委员会)")

    # 基础统计
    stats = {"n": len(results), "samples": results}
    if use_hy3:
        stds = [r["committee"]["committee_std"] for r in results if r.get("committee")]
        if stds:
            import statistics
            stats["std_mean"] = round(statistics.mean(stds), 2)
            stats["std_median"] = round(statistics.median(stds), 2)
            stats["std_min"] = round(min(stds), 2)
            stats["std_max"] = round(max(stds), 2)
            # 可靠性分布
            rel_counts = {"可信": 0, "存疑": 0, "需人工复核": 0}
            for r in results:
                c = r.get("committee")
                if c:
                    rel_counts[c["committee_reliability"]] = rel_counts.get(c["committee_reliability"], 0) + 1
            stats["reliability_distribution"] = rel_counts

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("=" * 50)
    print(f"委员会交叉验证样本数 n = {stats['n']}")
    if use_hy3 and stds:
        print(f"平均分歧度(std) = {stats['std_mean']} | 中位数 = {stats['std_median']} | 范围 [{stats['std_min']}, {stats['std_max']}]")
        print(f"可靠性分布: {stats['reliability_distribution']}")
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
