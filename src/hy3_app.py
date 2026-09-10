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
from typing import Any, Dict, List, Optional, Sequence

import config
import data_store
import retrieval

# 结构化输出：{answer, citations}
Output = Dict[str, Any]


# HTTP 连接池：跨多次调用复用底层 TCP/TLS 连接，显著降低并发推理场景的握手开销。
# 线程安全：requests.Session 非严格线程安全，但同一时刻对 session 的并发 post 由
# urllib3 连接池内部加锁处理，实测多线程（--workers 16）下可安全复用。
_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def call_hy3(messages: Sequence[Dict[str, str]], temperature: float = 0.2,
             max_tokens: int = 4096) -> Optional[str]:
    """通用 Hy3 / 混元 OpenAI 兼容调用。无 key 或失败返回 None。"""
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
        r = _get_session().post(url, headers=headers, json=payload, timeout=180)
        # 强制按 UTF-8 解码：部分 OpenAI 兼容网关返回 Content-Type 不带 charset，
        # requests 会按 latin-1 误解码，导致中文答案整段乱码（mojibake）。
        r.encoding = "utf-8"
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # 仅取正式回答 content。严禁回退到 reasoning_content：那是模型的思考草稿
        # （"我们需要输出…规则说…"），当答案用会污染标注池与评测，且思考过程里的
        # 关键词（规则/citations 等）会让规则评估器误打高分。content 为空 = 推理
        # 耗尽 max_tokens 被截断，应判失败并交由上层重试/降级。
        content = (msg.get("content") or "").strip()
        if not content:
            return None
        return content
    except KeyboardInterrupt:  # 允许用户 Ctrl+C 中断批量评测
        raise
    except Exception as e:  # 失败降级
        return f"[HY3_ERROR] {e}"


def _extract_json(text: str) -> str:
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        return text[s:e + 1]
    return text


def generate(sample: Dict[str, Any], use_rag: bool = True,
             pdf_pages: Optional[Any] = None,
             pdf_name: Optional[str] = None) -> Optional[Output]:
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
    recs = []  # [(year, rec)]，可能多年（对比类题目如「2022与2021两年」）
    if code:
        years = []
        if year and ind.get(f"{code}_{year}"):
            years.append(year)
        # 对比类题目（「2023与2022两年」）需要多年数据：主年份之外，
        # 补上输入中出现的其他年份。放在 if 之外，避免主年份命中时漏掉另一年。
        for y in re.findall(r"(?:19|20)\d{2}", sample["input"]):
            if y not in years and ind.get(f"{code}_{y}"):
                years.append(y)
        recs = [(y, ind[f"{code}_{y}"]) for y in years]
    has_ind = bool(use_rag and recs)
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
        assert code is not None  # has_ind 由 recs 推导，recs 仅当 code 非空才填充
        for y, rec in recs:
            ctx_parts.append("【真实指标表】（" + code + " " + y + "）："
                             + "；".join(f"{k}={v}" for k, v in rec.items()))
    if has_pdf:
        sel = retrieval.retrieve_pdf_pages(pdf_pages, sample["input"], top_k=3)
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = call_hy3(messages, max_tokens=8192)
    if not out or out.startswith("[HY3_ERROR]"):
        # content 为空多为推理过程耗尽 max_tokens 被截断，属偶发：重试一次
        out = call_hy3(messages, max_tokens=8192, temperature=0.6)
    if not out or out.startswith("[HY3_ERROR]"):
        return None

    parsed = _parse_hy3_json(out)
    if parsed is not None:
        return parsed
    if out.startswith("[HY3_ERROR]"):
        return None

    # 兜底：尝试从文本抽取 citations，避免引用维度直接归零
    cit = []
    m = re.search(r'"citations"\s*:\s*(\[.*\])', out, re.DOTALL)
    if m:
        try:
            cit = json.loads(m.group(1))
        except Exception:
            cit = []
    return {"answer": out, "citations": cit if isinstance(cit, list) else []}


def _strip_code_fences(text: str) -> str:
    """去掉模型可能包裹的 markdown 代码块标记（```json ... ```）。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉第一行 ```json 等
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def _parse_hy3_json(text: str) -> Optional[Output]:
    """多角度解析模型返回的 JSON，失败返回 None。

    模型输出可能：
      1) 纯 JSON 对象；
      2) 被 markdown 代码块包裹；
      3) JSON 外带少量说明文字；
      4) 返回的是字符串化的 JSON（双重序列化）。
    """
    candidates = [
        text,
        _strip_code_fences(text),
        _extract_json(text),
        _strip_code_fences(_extract_json(text)),
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            # 双重序列化兜底：偶尔模型把 JSON 字符串又包了一层字符串
            if isinstance(obj, str) and obj.strip().startswith("{"):
                obj = json.loads(obj.strip())
            if isinstance(obj, dict):
                # 确保至少返回 answer 字符串；citations 默认空列表
                answer = obj.get("answer") or ""
                citations = obj.get("citations") or []
                return {"answer": answer, "citations": citations}
        except Exception:
            continue
    return None
