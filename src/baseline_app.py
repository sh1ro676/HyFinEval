# -*- coding: utf-8 -*-
"""
基线应用（baseline_app）：不依赖 API key，基于本地真实指标做确定性抽取/判断。
作用：在没有 Hy3 key 时也能跑通"应用产生输出 -> 评估器打分"的完整闭环，便于先看效果。
拿到 Hy3 key 后，把 app.py 切换到 hy3_app.generate() 即可，评估层无需任何改动。

输出结构：{answer, citations}。answer 为多小节 Markdown，citations 为可验证引用列表。
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
            answer = (
                f"## 结论\n{field}为 **{val}** 元（{code} {year}年报口径）。\n\n"
                f"## 关键指标\n- {field}：{val} 元\n\n"
                f"## 简要分析\n该指标反映公司当期经营成果。本基线为确定性抽取，仅回显真实指标库中的数值，"
                f"不做主观解读；如需趋势判断，请结合多期数据与行业对比。\n\n"
                f"## 风险与提示\n数值以年报披露为准；投资决策需综合营收、现金流、负债等多维度，勿单凭单一指标。"
            )
            citations.append({"company": code, "year": year,
                              "field": field, "value": val, "source": "indicators"})
        else:
            answer = (
                f"## 结论\n未在 {code} {year} 指标库中找到【{field}】。\n\n"
                f"## 关键指标\n（无）\n\n"
                f"## 简要分析\n该字段可能未纳入本基线预置指标库，或需从年报原文提取；"
                f"启用 Hy3 后可基于上传 PDF 或更广数据源补全。\n\n"
                f"## 风险与提示\n具体数值须以公开年报原文为准，本基线不编造。"
            )

    elif subtask == "财务问答":
        val = data_store.get_true_value(code, year, field) if field else None
        if val is not None:
            answer = (
                f"## 结论\n关于『{field}』，{code} {year}年的值为 **{val}** 元。\n\n"
                f"## 关键指标\n- {field}：{val} 元\n\n"
                f"## 简要分析\n基线模型仅支持单指标确定性抽取，无法直接做跨期或衍生计算；"
                f"若需分析盈利能力与偿债压力，建议结合净利润率、资产负债率等指标进一步测算。\n\n"
                f"## 风险与提示\n以上为指标库回显值，未经模型推理；深度财务判断请以年报与专业分析为准。"
            )
            citations.append({"company": code, "year": year,
                              "field": field, "value": val, "source": "indicators"})
        else:
            answer = (
                "## 结论\n该问题涉及跨期对比或衍生计算，超出基线模型的确定性抽取能力。\n\n"
                "## 关键指标\n（基线不支持）\n\n"
                "## 简要分析\n基线仅回显单一真实指标，不支持跨期/衍生推算；"
                "启用 Hy3 后可基于指标表做公式推导与综合分析。\n\n"
                "## 风险与提示\n具体数值须以公开年报原文为准，本基线不编造。"
            )

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
        answer = (
            f"## 结论\n该公告属于【{atype}】类事项。\n\n"
            f"## 关键要素\n- 事项内容：{atype}\n"
            f"- 涉及主体：以公告披露为准\n"
            f"- 金额/比例：如公告含具体数字须以原文为准\n"
            f"- 生效时间：以审议/披露日为准\n"
            f"- 审议程序：董事会/股东大会等\n\n"
            f"## 简要分析\n以上为通用要素框架；本基线不读取公告原文，仅作结构提示。\n\n"
            f"## 风险与提示\n具体数字须以公告原文为准，本基线不编造。"
        )

    else:
        answer = "（基线模型不支持该子任务）"

    return {"answer": answer, "citations": citations}
