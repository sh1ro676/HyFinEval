# -*- coding: utf-8 -*-
"""stats_utils 统计函数测试。

重点验证平均秩（tie 修正）的 Spearman —— 本项目最大的并列数据正是此前
实现错误、后由 stats_utils 修正的地方（见模块 docstring）。用手算解析值
断言，确保并列秩与相关系数数学正确；边界返回 None。
"""
import pytest

import stats_utils as S


# ---------- average_rank ----------

def test_average_rank_no_tie():
    assert S.average_rank([10, 20, 30]) == [1.0, 2.0, 3.0]


def test_average_rank_with_tie():
    # 两个 20 并列占据秩位 2,3 -> 均值 2.5
    assert S.average_rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_average_rank_empty():
    assert S.average_rank([]) == []


def test_average_rank_all_tied():
    assert S.average_rank([7, 7, 7]) == [2.0, 2.0, 2.0]


# ---------- spearman ----------

def test_spearman_perfect_positive():
    assert S.spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_spearman_perfect_negative():
    assert S.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_tie_returns_zero():
    # a 秩 [1.5,1.5,3.5,3.5]，b 秩 [1.5,3.5,1.5,3.5]，两者偏差积和为零 -> 0
    assert S.spearman([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)


def test_spearman_short_returns_none():
    assert S.spearman([1], [1]) is None


def test_spearman_mismatched_len_returns_none():
    assert S.spearman([1, 2], [1, 2, 3]) is None


def test_spearman_none_input_returns_none():
    assert S.spearman(None, [1, 2, 3]) is None


# ---------- kendall_tau_b ----------

def test_kendall_monotonic_one():
    assert S.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_kendall_short_none():
    assert S.kendall_tau_b([1], [1]) is None


# ---------- quad_weighted_kappa ----------

def test_kappa_perfect_agreement_one():
    assert S.quad_weighted_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_kappa_constant_agreement_one():
    # 全一致（虽无区分度）在公式上返回 1.0（pe 饱和兜底）
    assert S.quad_weighted_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)
