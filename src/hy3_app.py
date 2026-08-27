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


def call_hy3(messages, temperature=0.2, max_tokens=1024):
    """通用 Hy3 / 混元 OpenAI 兼容调用。无 key 返回 None。"""
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
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # 失败降级
        return f"[HY3_ERROR] {e}"


def _extract_json(text):
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        return text[s:e + 1]
    return text


def generate(sample):
    """RAG 检索 + Hy3 生成。返回 {answer, citations} 或 None（降级用）。"""
    code, year, field = data_store.parse_input(sample["input"])
    ind = data_store.get_indicators()
    rec = ind.get(f"{code}_{year}")
    ctx = "；".join(f"{k}={v}" for k, v in rec.items()) if rec else "（无可用上下文）"

    system = (
        "你是金融数据分析助手。仅基于【真实财务指标】作答，必须输出严格 JSON："
        "{\"answer\": \"文本答案\", \"citations\": [{\"company\": \"代码\", \"year\": \"年份\", "
        "\"field\": \"指标名\", \"value\": 数值}]}。"
        "不得编造数据；指标不存在或无把握时明确说明需查年报原文。"
    )
    user = (
        f"真实财务指标（{code} {year}）：{ctx}\n\n"
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
        return {"answer": out, "citations": []}
