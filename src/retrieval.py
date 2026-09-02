# -*- coding: utf-8 -*-
"""检索模块：上传 PDF 的页面级检索（双源 RAG 之一）。

另一知识源为 data_store 中的指标真相库（indicators.json）。
"""
import re


def retrieve_pdf_pages(pages, query, top_k=3):
    """从已解析的 PDF 页中，按关键词/指标词重叠选出最相关的 top_k 页。

    pages: list[{"page":int,"text":str}]；返回 [(page, text), ...]
    """
    q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", query or ""))
    kw = ["营收", "净利润", "资产", "负债", "现金流", "毛利率", "净资产", "利润",
          "分红", "公告", "回购", "增持", "重组", "研发", "费用", "收入", "负债率"]
    scored = []
    for p in pages:
        text = p.get("text") or ""
        score = 0.0
        for t in q_tokens:
            if t and t in text:
                score += 1.0
        for k in kw:
            if k in text:
                score += 0.3
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored[:top_k] if s > 0]
    if not top:
        top = pages[:top_k]
    return [(p["page"], p["text"]) for p in top]
