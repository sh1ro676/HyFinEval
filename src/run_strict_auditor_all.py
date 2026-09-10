#!/usr/bin/env python3
"""严格审计员全面打分：28 条 label_pool + 122 条公告摘要 = 150 条。

用法：
  python src/run_strict_auditor_all.py --workers 10
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import evaluator
import rubric


def load_all_samples():
    """加载两部分数据并统一格式。"""
    items = []

    # ---- 28 条 label_pool（财报/问答，有人工标注）----
    with open("data_cache/label_pool.json", encoding="utf-8") as f:
        pool = json.load(f)
    for p in pool:
        items.append({
            "id": p["id"],
            "subtask": p.get("subtask", ""),
            "difficulty": p.get("difficulty", "中"),
            "sample": p,
            "output": {"answer": p.get("output", ""), "citations": p.get("citations", [])},
            "has_human_label": True,
        })

    # ---- 122 条公告摘要（无人工标注，有客观指标 fact_coverage）----
    with open("data_cache/ann_outputs.json", encoding="utf-8") as f:
        ann = json.load(f)
    for a in ann:
        # 把 citations 从顶层移到 output 里（ann_outputs 的 citations 在顶层）
        out = {"answer": a.get("output", ""), "citations": a.get("citations", [])}
        items.append({
            "id": a["id"],
            "subtask": a.get("subtask", "公告摘要"),
            "difficulty": a.get("difficulty", "中"),
            "sample": a,
            "output": out,
            "has_human_label": False,
        })

    return items


def call_strict_auditor(item):
    """对单条样本调用严格审计员角色。"""
    sid = item["id"]
    try:
        res = evaluator._call_judge_role(
            item["sample"], item["output"],
            "strict_auditor", rubric.JUDGE_PROMPT_STRICT_AUDITOR
        )
        if res and isinstance(res.get("overall"), (int, float)):
            return {
                "id": sid,
                "status": "ok",
                "overall": float(res["overall"]),
                "dimensions": res.get("dimensions", {}),
                "failure_mode": res.get("failure_mode", ""),
            }
        else:
            return {"id": sid, "status": "parse_fail", "overall": None}
    except Exception as e:
        return {"id": sid, "status": f"error:{e}", "overall": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    items = load_all_samples()
    print(f"总样本数: {len(items)} (人工标注={sum(1 for i in items if i['has_human_label'])}, 公告={sum(1 for i in items if not i['has_human_label'])})")

    results = []
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_strict_auditor, it): it for it in items}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                ok += 1
                print(f"  {r['id']} → {r['overall']:.1f}")
            else:
                print(f"  {r['id']} → {r['status']}")

    out = {
        "n_total": len(items),
        "n_ok": ok,
        "n_label_pool": sum(1 for i in items if i["has_human_label"]),
        "n_announcement": sum(1 for i in items if not i["has_human_label"]),
        "samples": results,
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "strict_auditor_all.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {ok}/{len(items)} 成功，结果写入 {path}")


if __name__ == "__main__":
    main()
