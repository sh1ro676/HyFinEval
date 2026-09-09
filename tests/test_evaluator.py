# -*- coding: utf-8 -*-
"""evaluator 核心评分逻辑测试。

覆盖：事实准确性（精确/偏差/错误）、反例荒谬量级拒斥（999999）、
完整性与格式维度基础行为。所有样本使用 indicators.json 中真实存在的
(600519, 2021, 扣非净利润=52581102656.24) 保证规则分支可达。
"""
import pytest

import evaluator

# 真实存在的数据：600519(贵州茅台) 2021 扣非净利润 = 52581102656.24
SAMPLE = {
    "subtask": "财报指标提取",
    "input": "贵州茅台（600519）2021年【扣非净利润】是多少？",
}
TRUE_VAL = 52581102656.24


def _eval(sample, answer, citations=None):
    out = {"answer": answer, "citations": citations or []}
    return evaluator._rule_evaluate(sample, out)


# ---------- 事实准确性（factual_accuracy） ----------

def test_factual_exact_value_scores_full():
    dims, _, _ = _eval(SAMPLE, f"扣非净利润为{TRUE_VAL:.2f}元。")
    assert dims["factual_accuracy"] == 1.0


def test_factual_small_error_halves():
    # 相对误差约 10% -> 落入 (1%, 20%] -> 0.5
    off = TRUE_VAL * 1.1
    dims, _, _ = _eval(SAMPLE, f"扣非净利润为{off:.2f}元。")
    assert dims["factual_accuracy"] == 0.5


def test_factual_wrong_value_zero():
    dims, _, _ = _eval(SAMPLE, "扣非净利润为10000000000元。")  # 差一个数量级
    assert dims["factual_accuracy"] == 0.0


def test_factual_no_number_zero():
    dims, _, _ = _eval(SAMPLE, "扣非净利润情况良好，符合预期。")
    assert dims["factual_accuracy"] == 0.0


def test_thousands_separator_not_fragmented():
    # "52,581,102,656.24" 去千分位后应视为单个数而非碎片
    dims, _, _ = _eval(SAMPLE, "扣非净利润为52,581,102,656.24元。")
    assert dims["factual_accuracy"] == 1.0


# ---------- 反例 / 荒谬量级拒斥（999999） ----------

ABSURD_SAMPLE = {
    "subtask": "财报指标提取",
    "input": "若某公司2021年扣非净利润达999999亿元，是否远超茅台？（600519）",
    "is_counterfeit": True,
}


def test_absurd_input_rejected_scores_full():
    dims, _, _ = _eval(ABSURD_SAMPLE, "该前提荒谬，999999亿元远超任何真实公司体量，不可比，不成立。")
    assert dims["factual_accuracy"] == 1.0
    assert dims["safety_no_hallucination"] == 1.0


def test_absurd_input_asserted_scores_zero():
    dims, _, _ = _eval(ABSURD_SAMPLE, "超过999999亿元，盈利远超茅台。")
    assert dims["factual_accuracy"] == 0.0
    assert dims["safety_no_hallucination"] == 0.0


# ---------- 完整性（completeness） ----------

def test_completeness_metric_present_is_full():
    dims, _, _ = _eval(SAMPLE, f"扣非净利润为{TRUE_VAL:.2f}元。")
    assert dims["completeness"] == 1.0


def test_completeness_metric_missing_is_zero():
    # 财报指标提取子任务：无输出数值 -> 完整性归零
    dims, _, _ = _eval(SAMPLE, "相关数据未在此披露。")
    assert dims["completeness"] == 0.0


# ---------- 引用可验证性（citation_verifiability） ----------

def test_verifiable_citation_boosted():
    out = {
        "answer": f"扣非净利润为{TRUE_VAL:.2f}元。",
        "citations": [{"company": "600519", "year": "2021", "field": "扣非净利润",
                       "value": TRUE_VAL, "source": "indicators"}],
    }
    dims, _, _ = evaluator._rule_evaluate(SAMPLE, out)
    # 全可核验引用 -> 0.4 + 0.6 = 1.0
    assert dims["citation_verifiability"] == 1.0


def test_no_citation_low():
    dims, _, _ = _eval(SAMPLE, f"扣非净利润为{TRUE_VAL:.2f}元。")
    assert dims["citation_verifiability"] == 0.1


# ---------- 公开入口 evaluate 返回结构 ----------

def test_evaluate_public_api_structure():
    out = {"answer": f"扣非净利润为{TRUE_VAL:.2f}元。", "citations": []}
    res = evaluator.evaluate(SAMPLE, out)
    assert set(res) >= {"dimensions", "overall", "failure_mode", "compliance_breaker", "output"}
    assert 0.0 <= res["overall"] <= 100.0
    assert res["compliance_breaker"] is None  # 正常输出不应触发合规熔断
