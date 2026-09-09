# -*- coding: utf-8 -*-
"""data_store 数据层检索测试：parse_input 解析 (code,year,field) 与真值检索。

使用 indicators.json 中真实存在的 (600519, 2021, 扣非净利润) 保证结果稳定。
"""
import data_store


def test_parse_input_brackets_code():
    # 代码写在中文括号（）内
    c, y, f = data_store.parse_input("贵州茅台（600519）2021年【扣非净利润】是多少？")
    assert c == "600519"
    assert y == "2021"
    assert f == "扣非净利润"


def test_parse_input_halfwidth_brackets():
    c, y, f = data_store.parse_input("五粮液(000858) 2022年【主营业务收入增长率】")
    assert c == "000858"
    assert y == "2022"
    assert f == "主营业务收入增长率"


def test_get_true_value_real_record():
    val = data_store.get_true_value("600519", "2021", "扣非净利润")
    assert val == 52581102656.24


def test_get_true_value_growth_rate_real():
    # 000858_2022 存在「主营业务收入增长率」字段
    val = data_store.get_true_value("000858", "2022", "主营业务收入增长率")
    assert isinstance(val, (int, float))


def test_get_true_value_unknown_company_none():
    assert data_store.get_true_value("999999", "2021", "扣非净利润") is None


def test_get_true_value_unknown_field_none():
    # 字段名不在指标库中
    assert data_store.get_true_value("600519", "2021", "不存在的指标XYZ") is None


def test_get_true_value_handles_suffix():
    # field 带「的数值」后缀做模糊匹配
    val = data_store.get_true_value("600519", "2021", "扣非净利润的数值")
    assert val == 52581102656.24
