# -*- coding: utf-8 -*-
"""
公告抽取共享工具（ann_utils）：跨 build_announcement_samples / score_announcement
复用的正则与数值归一化，避免同一组模式在多处重复定义、日后不一致。

共享内容：金额/比例/日期正则 + 数值归一化（去掉空白与千分位）。

注意：两个消费方对抽取结果的**语义**不同（一个保留单位用于事实核对、一个抽数值核心
用于匹配），因此各自的 extract_* 函数留在原模块，只有底层模式与归一化在此共享。
"""
import re

# ---- 关键数值模式（金额/比例/日期），与两消费方历史定义逐字符一致 ----
MONEY = re.compile(r"\d[\d,]*\.?\d*\s*(?:亿元|万元|元|股|份|手|万股|亿股)")
RATIO = re.compile(r"\d+(?:\.\d+)?\s*%")
DATE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}")
YMD_DASH = re.compile(r"\d{4}/\d{1,2}/\d{1,2}")


def norm(s: str) -> str:
    """归一化数值串：去掉空白与千分位（兼容原模块的 norm / normalize 两种命名）。"""
    return re.sub(r"[\s,]", "", s or "")


# 兼容旧命名：build_announcement_samples 曾用 normalize 调用
normalize = norm
