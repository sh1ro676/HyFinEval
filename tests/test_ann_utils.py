# -*- coding: utf-8 -*-
"""ann_utils 共享抽取工具测试。

验证公告金额/比例/日期正则与数值归一化：这是 build_announcement_samples 与
score_announcement 两链路的共享底层，正则回归能防止两处消费方行为漂移。
"""
import ann_utils as A


# ---------- 数值归一化 ----------

def test_norm_strips_thousands_and_space():
    assert A.norm("52,581,102,656.24") == "52581102656.24"
    assert A.norm(" 3.5 亿元 ") == "3.5亿元"


def test_norm_empty():
    assert A.norm("") == ""
    assert A.norm(None) == ""


# ---------- 金额 / 比例 ----------

def test_money_matches_unit_forms():
    text = "净利52581102656.24元，分红3.5亿元，另回购200万股"
    hits = A.MONEY.findall(text)
    assert "52581102656.24元" in hits
    assert "3.5亿元" in hits
    assert "200万股" in hits


def test_ratio_matches_percent():
    text = "营收增长22.81%，毛利率3.5%"
    assert A.RATIO.findall(text) == ["22.81%", "3.5%"]


# ---------- 日期 ----------

def test_date_variants_normalized():
    # DATE 正则不含「日」字，故 "2023年5月6日" 提取到 "2023年5月6"（供后续归一化补日）
    text = "披露日期2021-12-31；对比2022/03/01；另有2023年5月6日"
    hits = A.DATE.findall(text)
    assert "2021-12-31" in hits
    assert "2022/03/01" in hits
    assert "2023年5月6" in hits


def test_ymd_dash_slashes():
    assert A.YMD_DASH.findall("截至2022/12/31的数据") == ["2022/12/31"]
    # 连字符日期不应被斜杠版正则命中（二者口径不同）
    assert A.YMD_DASH.findall("截至2022-12-31") == []


# ---------- 命名兼容 ----------

def test_normalize_alias():
    assert A.normalize is A.norm
