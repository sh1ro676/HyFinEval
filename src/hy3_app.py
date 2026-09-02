# -*- coding: utf-8 -*-
"""
Hy3 应用层（hy3_app）：基于 Hy3 的真实应用生成 + 通用 Hy3 调用客户端。
A+B 特色：先检索 (公司,年份) 的真实指标作为上下文（RAG 思路），再让 Hy3 生成
结构化输出 {answer, citations}，实现"引用可验证"。
无 key 时 generate() 返回 None，由上层降级到 baseline_app。
"""
import json
import re
import requests
import config
import data_store


def call_hy3(messages, temperature=0.2, max_tokens=4096):
    """通用 Hy3 / 混元 OpenAI 兼容调用。无 key 返回 None。

    注意：hy3 是推理模型，会先生成 reasoning_content 再生成 content；
    若 max_tokens 太小，推理过程会耗尽额度导致 content 为空。
    这里默认 max_tokens=3072 给推理+正式回答留足空间，并做兼容兜底。
    """
    if not config.USE_HY3:
        return None
    url = config.HY3_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.HY3_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.HY3_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            # 推理模型兼容：content 为空时回退到思考过程，避免整条请求判失败
            content = (msg.get("reasoning_content") or "").strip()
        return content
    except Exception as e:  # 失败降级
        return f"[HY3_ERROR] {e}"


def _extract_json(text):
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        return text[s:e + 1]
    return text


def retrieve_pdf_pages(pages, query, top_k=3):
    """从已解析的 PDF 页中，按关键词/指标词重叠选出最相关的 top_k 页。

    pages: list[{"page":int,"text":str}]；返回 [(page, text), ...]
    """
    import re as _re
    q_tokens = set(_re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", query or ""))
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


def generate(sample, use_rag=True, pdf_pages=None, pdf_name=None):
    """RAG 检索 + Hy3 生成。返回 {answer, citations} 或 None（降级用）。

    上下文来源（可叠加）：
      - 指标表：use_rag=True 且 (code,year) 在 indicators.json 中（开卷，字段级可验证）。
      - 上传文档：pdf_pages 非空时，检索相关页注入，要求按页码引用（覆盖任意股票/年份）。
    两者皆无则闭卷，模型凭知识作答，数值须标注未经核实、不得编造。

    输出 answer 为多小节 Markdown（## 结论 / ## 关键指标 / ## 风险与关注 / ## 简要分析），
    杜绝一两句话敷衍；citations 含 field/value（指标）或 page/excerpt（文档）。
    """
    code, year, field = data_store.parse_input(sample["input"])
    ind = data_store.get_indicators()
    rec = ind.get(f"{code}_{year}")
    has_ind = bool(use_rag and rec)
    has_pdf = bool(pdf_pages)

    example = (
        '示例：{"answer":"## 结论\\n贵州茅台（600519）2021年扣非净利润为525.81亿元。'
        '\\n\\n## 关键指标\\n- 扣非净利润：52581102656.24元",'
        '"citations":[{"company":"600519","year":"2021","field":"扣非净利润",'
        '"value":52581102656.24,"source":"indicators"}]}'
    )

    # 组装上下文
    ctx_parts = []
    if has_ind:
        ctx_parts.append("【真实指标表】（" + code + " " + year + "）："
                         + "；".join(f"{k}={v}" for k, v in rec.items()))
    if has_pdf:
        sel = retrieve_pdf_pages(pdf_pages, sample["input"], top_k=3)
        pdf_ctx = "\n".join(f"[第{p}页] {t}" for p, t in sel)
        ctx_parts.append("【上传文档 " + (pdf_name or "PDF") + " 相关页】\n" + pdf_ctx)
    ctx = "\n\n".join(ctx_parts)

    src_note = []
    if has_ind:
        src_note.append("已提供【真实指标表】")
    if has_pdf:
        src_note.append("已提供【上传文档】")
    if not src_note:
        src_note.append("未提供任何真实数据（闭卷）")
    src_desc = "；".join(src_note)

    system = (
        "你是一名严谨的金融数据分析助手。" + src_desc + "。请输出一个 JSON 对象，结构：\n"
        "{\"answer\":\"多小节分析（Markdown，必须含：## 结论 / ## 关键指标 / "
        "## 风险与关注 / ## 简要分析 四个小节，有实质内容，禁止一两句话敷衍）\","
        "\"citations\":[{\"company\":\"代码\",\"year\":\"年份\",\"field\":\"指标名\",\"value\":数值,"
        "\"page\":页码,\"source\":\"indicators|pdf\",\"excerpt\":\"原文摘录\"}]}\n"
        "规则：\n"
        "1. 只输出一个 JSON 对象，不要任何额外解释文字。\n"
        "2. answer 必须分点、有实质内容；凡出现具体数值必须精确并列入 citations。\n"
        "3. 若提供了【真实指标表】，其中字段的具体数值必须精确抄录，不得自行改写；"
        "衍生指标用表中字段按标准公式展示推导，并仍给出表中真实值。\n"
        "4. 若提供了【上传文档】，回答优先基于文档，每条引用标注 page（页码）与 excerpt（原文摘录）；"
        "文档未覆盖的内容可补充说明，但须注明『据公开知识，未经文档核实』。\n"
        "5. 不得编造无法确认的具体数字；不确定写『需以原文为准』，对应 value 写 null。\n"
        "6. 逐条回应问题所有子问题，保持完整。\n"
        f"{example}"
    )

    if ctx:
        user = f"{ctx}\n\n问题：{sample['input']}\n\n请只输出 JSON。"
    else:
        user = (
            "当前为闭卷模式（未提供真实指标表或上传文档）。请基于你的金融知识作答：\n"
            "1. 对于金融概念、分析方法、分析框架、行业常识，直接用知识清晰作答，不要整体拒答；\n"
            "2. 对于具体财务数值：有把握可给并注明『据公开资料/记忆，未经指标库核实』，"
            "不确定写『需以年报原文为准』；\n"
            "3. 不得编造具体数字；若要求衍生指标，展示所用公式与基数。\n"
            f"问题：{sample['input']}\n\n请只输出 JSON。"
        )

    out = call_hy3([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    if not out or out.startswith("[HY3_ERROR]"):
        return None
    try:
        return json.loads(_extract_json(out))
    except Exception:
        # 兜底：JSON 解析失败时，尝试从文本抽取 citations，避免引用维度直接归零
        cit = []
        m = re.search(r'"citations"\s*:\s*(\[.*\])', out, re.DOTALL)
        if m:
            try:
                cit = json.loads(m.group(1))
            except Exception:
                cit = []
        return {"answer": out, "citations": cit if isinstance(cit, list) else []}
