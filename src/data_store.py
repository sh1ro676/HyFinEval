# -*- coding: utf-8 -*-
"""数据层：加载样本与真实指标，提供按 (代码,年份,字段) 的检索。"""
import json
import re
import config

_IND = None


def load_samples():
    with open(config.SAMPLES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_indicators():
    with open(config.INDICATORS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_indicators():
    global _IND
    if _IND is None:
        _IND = load_indicators()
    return _IND


def parse_input(text):
    """从样本 input 解析 (code, year, field)。"""
    code = None
    m = re.search(r"[（(](\d{6})[）)]", text)
    if m:
        code = m.group(1)
    year = None
    m = re.search(r"(\d{4})\s*年", text)
    if m:
        year = m.group(1)
    field = None
    m = re.search(r"【([^】]+)】", text)
    if m:
        field = m.group(1)
    return code, year, field


def get_true_value(code, year, field):
    """取真实指标值。field 可能带『的数值』等后缀，做模糊匹配。"""
    ind = get_indicators()
    rec = ind.get(f"{code}_{year}")
    if rec is None:
        return None
    if not field:
        return None
    field_clean = field.replace("的数值", "").replace("的数值？", "").strip()
    if field_clean in rec:
        return rec[field_clean]
    for k, v in rec.items():
        if field_clean and field_clean in k:
            return v
    return None


def company_name(code):
    names = {
        "600519": "贵州茅台", "000858": "五粮液", "300750": "宁德时代",
        "002594": "比亚迪", "601318": "中国平安", "000001": "平安银行",
        "600036": "招商银行", "600276": "恒瑞医药",
    }
    return names.get(code, code)
