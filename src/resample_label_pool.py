# -*- coding: utf-8 -*-
"""定向重生成标注池中的指定条目（不触碰其他条目）。

背景：Hy3 偶发 content 为空（推理耗尽 max_tokens），旧版 call_hy3 会把
reasoning_content（思考草稿）当答案写进标注池。修复生成侧后，用本脚本
只重生成受影响的 id，原位替换 output/citations/auto_score，其余条目
（含已人工标注的条目）保持原样，保证标注与输出一一对应。

用法：
  python src/resample_label_pool.py --id FIN-019
  python src/resample_label_pool.py --id FIN-019 --id FIN-005 --id FIN-059
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import data_store
import hy3_app
import evaluator

POOL = os.path.join(config.ROOT_DIR, "data_cache", "label_pool.json")

# 与 hy3_app 保持一致的泄漏检测：新生成内容不得再含思考草稿特征
_ECHO_MARKERS = ("我们需要输出一个JSON对象", "只输出一个 JSON 对象",
                 "只输出一个JSON对象", "请只输出 JSON", "禁止一两句话敷衍")


def is_echo(text):
    t = str(text or "")
    return sum(1 for m in _ECHO_MARKERS if m in t) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", required=True)
    args = ap.parse_args()
    ids = set(args.id)

    pool = json.load(open(POOL, encoding="utf-8"))
    by_id = {it["id"]: (i, it) for i, it in enumerate(pool)}
    samples = {s["id"]: s for s in data_store.load_samples()}

    for rid in sorted(ids):
        if rid not in by_id or rid not in samples:
            print(f"  {rid} 不在池中或样本集，跳过")
            continue
        s = samples[rid]
        out = hy3_app.generate(s, use_rag=True)
        if not out or not str(out.get("answer", "")).strip():
            print(f"  {rid} 重生成失败（Hy3 不可用或仍截断），保持原条目不动")
            continue
        if is_echo(out.get("answer", "")):
            print(f"  {rid} 新输出仍疑似思考草稿，保持原条目不动")
            continue
        ev = evaluator.evaluate(s, out)
        i, old = by_id[rid]
        new_it = {
            "id": rid,
            "subtask": s.get("subtask"),
            "difficulty": s.get("difficulty"),
            "input": s.get("input"),
            "output": out.get("answer"),
            "citations": out.get("citations", []),
            "auto_score": round(ev["overall"], 1),
        }
        print(f"  {rid}: auto {old.get('auto_score')} -> {new_it['auto_score']}"
              f"  answer_len {len(str(old.get('output') or ''))} -> {len(new_it['output'])}")
        pool[i] = new_it

    with open(POOL, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"已写回 {POOL}（其余条目未改动）")


if __name__ == "__main__":
    main()
