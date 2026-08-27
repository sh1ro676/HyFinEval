# -*- coding: utf-8 -*-
"""
基线应用（baseline_app）：不依赖 API key，基于本地真实指标做确定性抽取/判断。
作用：在没有 Hy3 key 时也能跑通"应用产生输出 -> 评估器打分"的完整闭环，便于先看效果。
拿到 Hy3 key 后，把 app.py 切换到 hy3_app.generate() 即可，评估层无需任何改动。
"""
import re
import data_store


def _extract_first_number(text):
    m = re.search(r"[-+]?\d*\.?\d+", str(text).replace(",", ""))
    return float(m.group()) if m else None


def baseline_generate(sample):
    inp = sample["input"]
    subtask = sample["subtask"]
    code, year, field = data_store.parse_input(inp)
    citations = []
    answer = ""

    if subtask == "财报指标提取":
        val = data_store.get_true_value(code, year, field)
        if val is not None:
            answer = f"{field}：{val}（单位：元，来源：{code} {year}年报口径）"
            citations.append({"company": code, "year": year,
                              "field": field, "value": val})
        else:
            answer = f"未在 {code} {year} 数据中找到【{field}】。"

    elif subtask == "财务问答":
        val = data_store.get_true_value(code, year, field) if field else None
        if val is not None:
            answer = f"{field}为 {val}。"
            citations.append({"company": code, "year": year,
                              "field": field, "value": val})
        else:
            answer = ("该问题涉及跨期对比或衍生计算，基线模型仅支持单指标直接提取；"
                      "建议基于公开年报数据进一步计算。")

    elif subtask == "公告摘要":
        t = inp
        if "分红" in t or "利润分配" in t or "回报股东" in t:
            atype = "利润分配/分红"
        elif "增持" in t:
            atype = "股东增持"
        elif "回购" in t:
            atype = "股份回购"
        elif "重组" in t or "并购" in t or "定增" in t:
            atype = "资产重组/再融资"
        elif "股权激励" in t or "期权" in t or "限制性股票" in t:
            atype = "股权激励"
        else:
            atype = "其他重大事项"
        answer = (f"该公告属于【{atype}】类。关键要素通常包括：事项内容、涉及主体、"
                  f"金额/比例（如有）、生效时间、审议程序等；具体数字须以公告原文为准，本基线不编造。")

    else:
        answer = "（基线模型不支持该子任务）"

    return {"answer": answer, "citations": citations}
