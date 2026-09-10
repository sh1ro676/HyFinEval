#!/usr/bin/env python3
"""公告摘要专用评估器：新增 info_coverage（信息覆盖度）维度。

基于 ann_scores.json 中的客观指标（fact_coverage / hallucination_rate /
substantive_lines / hedge_count）计算 info_coverage，验证其能否区分
高/低覆盖率组——解决严格审计员在公告摘要上"覆盖率区分度不足"的问题。

用法：
  python src/ann_evaluator.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import stats_utils


def compute_info_coverage(score_row: dict) -> float:
    """基于客观指标计算 info_coverage（0-1 连续分）。

    设计逻辑：
    - 核心信号是 fact_coverage（模型输出覆盖了多少 ground truth 事实）
    - 辅以 anti-hallucination（不编造）、substantiveness（有实质内容）、
      anti-hedge（不回避）三个修正项
    - 闭卷样本 fact_coverage 天然低，但 info_coverage 作为"信息覆盖度"
      衡量的是"输出中包含了多少信息"，闭卷低是预期行为
    """
    fc = score_row.get("fact_coverage", 0.0)
    if fc is None:
        fc = 0.0

    # 幻觉惩罚（有幻觉扣分）
    hr = score_row.get("hallucination_rate")
    if hr is None:
        halluc_penalty = 0.0
    else:
        halluc_penalty = min(hr * 0.5, 0.25)

    # 实质内容奖励（有实质内容加分，无内容扣分）
    sl = score_row.get("substantive_lines", 0)
    if sl is None:
        sl = 0
    # 公告摘要通常 5-20 行实质内容，<3 行视为空洞，>15 行视为充实
    if sl >= 15:
        substance = 0.25
    elif sl >= 5:
        substance = 0.15
    elif sl >= 3:
        substance = 0.05
    else:
        substance = 0.0

    # 免责/回避话术惩罚（ hedge_count 高说明回避关键信息）
    hc = score_row.get("hedge_count", 0)
    if hc is None:
        hc = 0
    if hc >= 20:
        hedge_penalty = 0.25
    elif hc >= 10:
        hedge_penalty = 0.15
    elif hc >= 5:
        hedge_penalty = 0.05
    else:
        hedge_penalty = 0.0

    # 信息覆盖度 = 核心覆盖率 - 幻觉惩罚 + 实质奖励 - 回避惩罚
    raw = fc - halluc_penalty + substance - hedge_penalty
    return max(0.0, min(1.0, raw))


def evaluate_announcement(ann_output: dict, ann_score: dict) -> dict:
    """对单条公告摘要做完整评估（七维 + info_coverage）。"""
    # 七维规则分：复用现有 evaluator 的逻辑太重量级，这里简化为
    # 直接用 ann_scores.json 的客观指标映射到七维
    # 实际上公告摘要的七维评估与财报/问答不同，这里做适配映射

    out_text = ann_output.get("output", "")
    cit = ann_output.get("citations", [])
    mode = ann_output.get("mode", "闭卷")

    # ---- 维度1: 事实准确性 ----
    # 公告摘要中，fact_coverage 直接反映事实准确性
    fc = ann_score.get("fact_coverage", 0.0) or 0.0
    dim_factual = min(1.0, fc * 2.0)  # coverage 0.5 即满分，线性映射

    # ---- 维度2: 引用可验证性 ----
    # 公告摘要有 citations 即可，但闭卷通常无引用
    if isinstance(cit, list) and len(cit) > 0:
        dim_citation = 0.8
    elif "以公告原文为准" in out_text or "以原文为准" in out_text:
        dim_citation = 0.5
    else:
        dim_citation = 0.3 if mode == "闭卷" else 0.2

    # ---- 维度3: 完整性 ----
    # 用 coverage_amount + coverage_ratio + coverage_date 综合
    ca = ann_score.get("coverage_amount", 0.0) or 0.0
    cr = ann_score.get("coverage_ratio", 0.0) or 0.0
    cd = ann_score.get("coverage_date", 0.0) or 0.0
    dim_complete = min(1.0, (ca + cr + cd) / 3.0 * 2.0)

    # ---- 维度4: 格式/可解释 ----
    # 看是否有结构化输出（标题、段落）
    has_structure = bool(out_text and ("##" in out_text or "**" in out_text or "\n\n" in out_text))
    dim_format = 0.8 if has_structure else 0.5

    # ---- 维度5: 安全/抗幻觉 ----
    hr = ann_score.get("hallucination_rate")
    if hr is None:
        dim_safety = 0.8  # 未知时中性分
    else:
        dim_safety = max(0.0, 1.0 - hr * 2.0)

    # ---- 维度6: 数值计算 ----
    # 公告摘要不涉及计算，给中性分
    dim_computation = 0.6

    # ---- 维度7: 不确定性校准 ----
    # 闭卷下说"以原文为准"是恰当回避；开卷下应给出具体信息
    hedge = ann_score.get("hedge_count", 0) or 0
    if mode == "闭卷":
        if hedge <= 5:
            dim_calib = 0.8
        elif hedge <= 15:
            dim_calib = 0.5
        else:
            dim_calib = 0.3
    else:
        # 开卷下应少回避
        if hedge <= 3:
            dim_calib = 0.9
        elif hedge <= 10:
            dim_calib = 0.6
        else:
            dim_calib = 0.3

    # ---- 维度8: 信息覆盖度（新增）----
    dim_info_coverage = compute_info_coverage(ann_score)

    dims = {
        "factual_accuracy": round(dim_factual, 3),
        "citation_verifiability": round(dim_citation, 3),
        "completeness": round(dim_complete, 3),
        "format": round(dim_format, 3),
        "safety_no_hallucination": round(dim_safety, 3),
        "computation": round(dim_computation, 3),
        "calibration": round(dim_calib, 3),
        "info_coverage": round(dim_info_coverage, 3),
    }

    # 加权总分（含 info_coverage）
    weights = dict(config.DIMENSION_WEIGHTS)
    # 把 completeness 权重分给 info_coverage（因为公告摘要中 completeness 已映射到 coverage）
    # 实际上更合理的做法：给 info_coverage 分配独立权重
    # 这里暂时把 completeness 的权重归零，info_coverage 给 0.15
    weights["completeness"] = 0.0
    weights["info_coverage"] = 0.15
    # 其余权重按比例归一化
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    overall = sum(dims[k] * weights.get(k, 0.0) for k in dims)
    overall = round(overall * 100, 1)

    # 再算一个"不含 info_coverage"的版本作为对照
    weights_no_info = dict(config.DIMENSION_WEIGHTS)
    weights_no_info["completeness"] = 0.12  # 恢复原来的 completeness
    total2 = sum(weights_no_info.values())
    weights_no_info = {k: v / total2 for k, v in weights_no_info.items()}
    overall_no_info = sum(dims[k] * weights_no_info.get(k, 0.0) for k in dims if k in weights_no_info)
    overall_no_info = round(overall_no_info * 100, 1)

    return {
        "id": ann_output["id"],
        "mode": mode,
        "type": ann_output.get("type", ""),
        "dimensions": dims,
        "overall": overall,
        "overall_no_info_coverage": overall_no_info,
        "fact_coverage": fc,
        "info_coverage": dim_info_coverage,
    }


def main():
    # ---- 加载数据 ----
    with open("data_cache/ann_outputs.json", encoding="utf-8") as f:
        ann_outputs = json.load(f)
    with open("data_cache/ann_scores.json", encoding="utf-8") as f:
        ann_scores = json.load(f)

    scores_by_id = {s["id"]: s for s in ann_scores}
    outputs_by_id = {o["id"]: o for o in ann_outputs}

    results = []
    for oid, out in outputs_by_id.items():
        score = scores_by_id.get(oid)
        if not score:
            continue
        ev = evaluate_announcement(out, score)
        results.append(ev)

    # ---- 分组对比 ----
    high = [r for r in results if r["fact_coverage"] > 0.3]
    low = [r for r in results if r["fact_coverage"] <= 0.3]

    report = {
        "n_total": len(results),
        "with_info_coverage": {
            "high_coverage_mean": round(sum(r["overall"] for r in high) / len(high), 1) if high else None,
            "low_coverage_mean": round(sum(r["overall"] for r in low) / len(low), 1) if low else None,
            "diff": round(sum(r["overall"] for r in high) / len(high) - sum(r["overall"] for r in low) / len(low), 1) if high and low else None,
        },
        "without_info_coverage": {
            "high_coverage_mean": round(sum(r["overall_no_info_coverage"] for r in high) / len(high), 1) if high else None,
            "low_coverage_mean": round(sum(r["overall_no_info_coverage"] for r in low) / len(low), 1) if low else None,
            "diff": round(sum(r["overall_no_info_coverage"] for r in high) / len(high) - sum(r["overall_no_info_coverage"] for r in low) / len(low), 1) if high and low else None,
        },
        "info_coverage_correlation": {
            "info_coverage_vs_fact_coverage": round(stats_utils.spearman(
                [r["info_coverage"] for r in results],
                [r["fact_coverage"] for r in results]
            ), 3) if len(results) >= 2 else None,
            "overall_with_vs_fact_coverage": round(stats_utils.spearman(
                [r["overall"] for r in results],
                [r["fact_coverage"] for r in results]
            ), 3) if len(results) >= 2 else None,
            "overall_without_vs_fact_coverage": round(stats_utils.spearman(
                [r["overall_no_info_coverage"] for r in results],
                [r["fact_coverage"] for r in results]
            ), 3) if len(results) >= 2 else None,
        },
        "samples": results,
    }

    out_path = "src/results/ann_evaluator.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("公告摘要专用评估器（含 info_coverage 维度）")
    print("=" * 60)
    print(f"\n总样本数: {len(results)}")
    print(f"高覆盖率组 (>0.3): {len(high)} 条 | 低覆盖率组 (≤0.3): {len(low)} 条")

    print(f"\n【含 info_coverage 维度】")
    w = report["with_info_coverage"]
    print(f"  高覆盖率均分: {w['high_coverage_mean']}")
    print(f"  低覆盖率均分: {w['low_coverage_mean']}")
    print(f"  区分度 (差值): {w['diff']}")

    print(f"\n【不含 info_coverage 维度（对照）】")
    wo = report["without_info_coverage"]
    print(f"  高覆盖率均分: {wo['high_coverage_mean']}")
    print(f"  低覆盖率均分: {wo['low_coverage_mean']}")
    print(f"  区分度 (差值): {wo['diff']}")

    print(f"\n【相关性】")
    for k, v in report["info_coverage_correlation"].items():
        print(f"  {k}: {v}")

    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
