# -*- coding: utf-8 -*-
"""
单条演示：选一条样本，展示 应用输出 + 各维度评分，直观看效果。
用法：python run_demo.py [样本ID]，缺省演示 FIN-001（财报指标提取）与 FIN-044（反例）。
"""
import sys
import json
import data_store
import baseline_app
import hy3_app
import evaluator
import config


def demo(sample_id):
    s = next((x for x in data_store.load_samples() if x["id"] == sample_id), None)
    if not s:
        print(f"未找到 {sample_id}")
        return
    out = hy3_app.generate(s) if config.USE_HY3 else None
    if not out:
        out = baseline_app.baseline_generate(s)
    ev = evaluator.evaluate(s, out)
    print("─" * 50)
    print(f"样本 {s['id']} | {s['subtask']} | 难度 {s['difficulty']} | 反例 {s.get('is_counterfeit')}")
    print(f"输入：{s['input']}")
    print(f"应用输出：{out.get('answer')}")
    print(f"引用：{json.dumps(out.get('citations'), ensure_ascii=False)}")
    print("评估维度：")
    for k, v in ev["dimensions"].items():
        print(f"  - {k}: {v}")
    print(f"综合分：{ev['overall']} | 失败模式：{ev['failure_mode']}")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["FIN-001", "FIN-044", "FIN-047", "FIN-059"]
    for i in ids:
        demo(i)
