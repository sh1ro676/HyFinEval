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


def call_hy3(messages, temperature=0.2, max_tokens=3072):
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


def generate(sample, use_rag=True):
    """RAG 检索 + Hy3 生成。返回 {answer, citations} 或 None（降级用）。

    use_rag=True（开卷）：检索 (公司,年份) 真实指标作为上下文，考"引用可验证"。
    use_rag=False（闭卷）：不注入任何指标表，纯考模型自身金融知识，用于证伪
        computation / factual_accuracy 维度的判别力（剥离 RAG 后是否仍有效）。

    2026-08-31（二轮）提示词优化：要求精确抄录、强制 citations、逐条完整性、
    衍生指标展示公式、不确定即声明查原文——目标是在现有 rubric 下合法提分。
    """
    code, year, field = data_store.parse_input(sample["input"])
    ind = data_store.get_indicators()
    rec = ind.get(f"{code}_{year}")

    example = (
        '示例：{"answer":"贵州茅台（600519）2021年扣非净利润为525.81亿元（52581102656.24元）。",'
        '"citations":[{"company":"600519","year":"2021","field":"扣非净利润","value":52581102656.24}]}'
    )

    if use_rag and rec:
        ctx = "；".join(f"{k}={v}" for k, v in rec.items())
        system = (
            "你是一名严谨的金融数据分析助手。请【严格基于】下方提供的真实财务指标作答，并遵守：\n"
            "1. 只输出一个 JSON 对象，不要任何额外解释文字。\n"
            "2. 结构：{\"answer\":\"一句话结论（含关键数值与单位）\","
            "\"citations\":[{\"company\":\"股票代码\",\"year\":\"年份\",\"field\":\"指标名\",\"value\":数值}]}。\n"
            "3. answer 中的数值必须从下方指标表【精确抄录】，不要自行计算或改写；"
            "若问题要求衍生指标或解释计算含义，用表中字段按标准公式展示推导，并仍给出表中真实值。\n"
            "4. citations 必须列出 answer 引用的每一个字段，field 严格使用指标表中的字段名，"
            "value 为该字段在表中的真实数值。\n"
            "5. 逐条回应问题的所有子问题，保持完整；指标不存在或无把握时说明需查年报原文。\n"
            f"{example}"
        )
        user = (
            f"真实财务指标（{code} {year}）：{ctx}\n\n"
            f"问题：{sample['input']}\n\n请只输出 JSON。"
        )
    else:
        system = (
            "你是一名严谨的金融数据分析助手。基于你的金融知识作答，并遵守：\n"
            "1. 只输出一个 JSON 对象，不要任何额外解释文字。\n"
            "2. 结构：{\"answer\":\"结论（含数值与单位）\","
            "\"citations\":[{\"company\":\"代码\",\"year\":\"年份\",\"field\":\"指标名\",\"value\":数值}]}。\n"
            "3. citations 列出你引用/依据的来源字段，value 为该指标真实数值。\n"
            "4. 若问题要求衍生指标，展示所用公式与基数。\n"
            "5. 若你不确定某数据或它不在你的知识范围内，明确说明需查年报原文，value 可写 null，不得编造具体数字。\n"
            "6. 逐条回应所有子问题。\n"
            f"{example}"
        )
        user = (
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
