# -*- coding: utf-8 -*-
"""compliance 合规熔断层测试。

三条红线各自触发一次 + 不误伤正常/免责输出：
  1) 编造未披露数字：有具体数值、引用/事实维度低、无审慎话术；
  2) 伪造引用：citations 非空但全部无法核验；
  3) 违规荐股/承诺收益：命中买入/必涨/保本等话术。
"""
import compliance

SAMPLE = {"subtask": "财报指标提取"}
OK_DIMS = {"citation_verifiability": 0.9, "factual_accuracy": 1.0}
LOW_DIMS = {"citation_verifiability": 0.1, "factual_accuracy": 0.0}
CAP = compliance.COMPLIANCE_CAP


def test_advice_redline_caps():
    out = {"answer": "建议买入并持有该股，必涨。", "citations": []}
    overall, reason = compliance.circuit_breaker(SAMPLE, out, OK_DIMS, 80.0)
    assert overall == CAP
    assert reason and "违规荐股" in reason


def test_fabricated_number_redline_caps():
    out = {"answer": "该公司2021年净利润高达500亿元。", "citations": []}
    overall, reason = compliance.circuit_breaker(SAMPLE, out, LOW_DIMS, 80.0)
    assert overall == CAP
    assert reason and "编造未披露数字" in reason


def test_hedged_statement_not_breached():
    # 有免责话术时红线1不应误伤
    out = {"answer": "据公开资料未经核实，该公司或披露净利约500亿元。", "citations": []}
    overall, reason = compliance.circuit_breaker(SAMPLE, out, LOW_DIMS, 80.0)
    assert overall == 80.0
    assert reason is None


def test_fake_citation_redline_caps():
    out = {"answer": "数值为500亿元。",
           "citations": [{"company": "600519", "year": "2021",
                          "field": "不存在的指标XYZ", "source": "臆造"}]}
    overall, reason = compliance.circuit_breaker(SAMPLE, out, OK_DIMS, 80.0)
    assert overall == CAP
    assert reason and "伪造引用" in reason


def test_normal_output_not_breached():
    out = {"answer": "扣非净利润为52581102656.24元。",
           "citations": [{"company": "600519", "year": "2021",
                          "field": "扣非净利润", "value": 52581102656.24,
                          "source": "indicators"}]}
    overall, reason = compliance.circuit_breaker(SAMPLE, out, OK_DIMS, 80.0)
    assert overall == 80.0
    assert reason is None


def test_breach_caps_high_score_only():
    # 即使 overall 很低（如 10），也不应再被压低 —— cap 用 min()
    out = {"answer": "建议买入。", "citations": []}
    overall, _ = compliance.circuit_breaker(SAMPLE, out, OK_DIMS, 10.0)
    assert overall == 10.0  # min(10, 40) = 10，不反向抬高
