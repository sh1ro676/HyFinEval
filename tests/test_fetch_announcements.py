# -*- coding: utf-8 -*-
"""fetch_announcements.exchange_of 交易所判定测试。

历史实现 "6 开头=沪否则深" 会把 9 开头沪市 B 股误判为深市、也无法明确区分
科创/创业等。本测试锁定修正后的前缀映射正确性（沪：600/601/603/605/688/689/900；
深：000/001/002/003/300/301/200），并确保历史 6 位 A 股行为不回退。
"""
import pytest

import fetch_announcements as F


@pytest.mark.parametrize("code,expected", [
    # 沪市主板 / 科创板 / 存托凭证 / 沪 B
    ("600519", "sse"), ("601318", "sse"), ("603288", "sse"),
    ("605499", "sse"), ("688981", "sse"), ("689009", "sse"),
    ("900901", "sse"),                      # 沪市 B 股（历史会误判为深）
    # 深市主板 / 中小板(并入主板) / 创业板 / 深 B
    ("000858", "szse"), ("001979", "szse"), ("002594", "szse"),
    ("003816", "szse"), ("300750", "szse"), ("301236", "szse"),
    ("200596", "szse"),                     # 深市 B 股
])
def test_exchange_of_prefix_mapping(code, expected):
    assert F.exchange_of(code) == expected


def test_exchange_of_fallback_for_unknown_prefix():
    # 未知前缀按首位数字兜底：6xx 归沪（历史行为），其余归深
    assert F.exchange_of("699999") == "sse"
    assert F.exchange_of("456789") == "szse"


def test_exchange_of_numeric_input():
    # 支持数字入参（内部做 str 转换）
    assert F.exchange_of(600519) == "sse"
    assert F.exchange_of(300750) == "szse"
