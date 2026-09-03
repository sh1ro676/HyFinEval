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


_FIELDS = None


def _known_fields():
    """指标库中出现过的指标名，长名优先（避免「净利润」抢掉「净利润增长率」）。"""
    global _FIELDS
    if _FIELDS is None:
        names = set()
        for rec in get_indicators().values():
            if isinstance(rec, dict):
                names.update(rec.keys())
        _FIELDS = sorted(names, key=lambda x: (-len(x), x))
    return _FIELDS


def parse_input(text):
    """从样本 input 解析 (code, year, field)。

    三类兜底（针对自然语言提问，非【】模板题）：
      - year：如「2022与2021两年」正则取不到，改为扫描所有合理年份，
        取指标库中确实存在的（避免 year=None 导致整条拿不到真值）。
      - field：无【】标记时，用指标库已知字段名做名词匹配。
      - 选择性表述（如「销售毛利率（或主营业务收入增长率）」）不强行匹配 field：
        答其中任一都算对，套真值反而会误杀正确答案，故保守留空。
    """
    code = None
    m = re.search(r"[（(](\d{4,6})[）)]", text)
    if m:
        code = m.group(1)

    year = None
    m = re.search(r"(\d{4})\s*年", text)
    if m:
        year = m.group(1)
    else:
        ind = get_indicators()
        for y in re.findall(r"(?:19|20)\d{2}", text):
            if code and ind.get(f"{code}_{y}"):
                year = y
                break

    field = None
    m = re.search(r"【([^】]+)】", text)
    if m:
        field = m.group(1)
    elif not re.search(r"[（(]\s*或[^）)]*[）)]", text):
        for name in _known_fields():
            if name in text:
                field = name
                break
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
        "600036": "招商银行", "600276": "恒瑞医药", "000333": "美的集团",
        "00700": "腾讯控股",
    }
    return names.get(code, code)
