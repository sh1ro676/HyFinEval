# -*- coding: utf-8 -*-
"""
评估引擎（evaluator）：对 (样本, 模型输出) 按 rubric 打分。
主评估器 = 规则打分（可复现、零成本、A+B 引用可验证硬维度）；
若配置了 Hy3 key，可启用 hy3_judge 做 LLM 交叉验证（--use-hy3-judge）。
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

import config
import data_store
import rubric
import hy3_app
import compliance

# 样本与输出均为 dict（结构见 samples.json），评分维度集为 str->float
Sample = Dict[str, Any]
Output = Dict[str, Any]


def _extract_numbers(text: Any) -> List[float]:
    # 先去千分位逗号，避免 "74,752,564,425.52" 被拆成多个碎片数导致数值匹配错位
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", str(text).replace(",", ""))]


# 反例样本（荒谬量级/量纲比较）中，模型是否"正确拒斥"而非"断言成立"
_ABSURD_REJECT = ("量纲", "不可比", "无法比较", "不成立", "远低于", "远大于",
                  "未超过", "不能比较", "不可直接比较", "没有意义", "不可比")
def _asserts_absurd_exceeds(text: Any) -> bool:
    t = str(text).replace(" ", "")
    return ("超过999999" in t) or ("超过了999999" in t) or ("达到999999" in t) or ("达到了999999" in t)


# 免责话术：审慎表达的正确形式，但也可能被用作"通篇正确的废话"填充篇幅
_HEDGE_PAT = re.compile(
    r"以原文为准|以公告原文|不编造|需以|无法精确|无法获取|未提供|请自行|"
    r"据公开知识|未经|须以|应由|待核实|以正式")

# 结构性小节（与 hy3_app 生成模板对应）
_SECTIONS = ("结论", "关键指标", "风险", "分析")


def _substantive_lines(out_text: Any, min_core: int = 8) -> int:
    """统计「有信息量」的正文行数：剔除标题、空行、纯免责话术行。

    用于区分「有实质内容且审慎」与「通篇免责话术」——后者在人类判断中
    属于低质量输出，但纯关键词打分会给满分。
    """
    n = 0
    for ln in str(out_text).splitlines():
        s = ln.strip().lstrip("-*·").lstrip()
        s = re.sub(r"^\d+[.、)]\s*", "", s).strip()
        if len(s) < 6:
            continue
        core = _HEDGE_PAT.sub("", s)
        core = re.sub(r"[\s，。、：:；;（）()【】\[\]]", "", core)
        if len(core) >= min_core:
            n += 1
    return n


def _format_score(output: Output, out_text: str) -> float:
    """格式规范性：结构覆盖度 × 实质内容，替代原「泛词命中即满分」实现。

    原实现判定词含「指标」「根据」等中文财报回答几乎必然出现的词，
    实测 28 条标注样本 28/28 满分（零方差 → 零判别力）。
    """
    if not out_text.strip():
        return 0.2
    if not (isinstance(output, dict) and output.get("answer")):
        return 0.6

    heads = re.findall(r"^#{1,4}\s*(.+?)\s*$", out_text, re.M)
    hit = sum(1 for s in _SECTIONS if any(s in h for h in heads))
    if hit >= 4:
        score = 1.0
    elif hit == 3:
        score = 0.85
    elif hit == 2:
        score = 0.7
    elif hit == 1:
        score = 0.5
    else:
        score = 0.4

    # 空壳惩罚：结构齐了但正文没有实质内容，说明是模板套话
    if _substantive_lines(out_text) < 3:
        score = min(score, 0.5)
    return score


def _rule_evaluate(sample: Sample, output: Output) -> Tuple[Dict[str, float], float, str]:
    out_text = output.get("answer", "") if isinstance(output, dict) else str(output)
    cit = output.get("citations", []) if isinstance(output, dict) else []
    subtask = sample["subtask"]
    inp = sample["input"]
    code, year, field = data_store.parse_input(inp)
    true_val = data_store.get_true_value(code, year, field) if field else None
    counterfeit = bool(sample.get("is_counterfeit", False))
    dims = {}
    # 缓存整条输出提取的数值，供 factual/safety/computation 等多维度复用，
    # 避免同一段文本被多次重复正则扫描（每样本省 2 次全量正则）。
    out_nums = _extract_numbers(out_text)

    # 跨期/两年对比检测（第 6 条维度打磨）：题目问「Y2 与 Y1 两年对比」时，
    # 需要核对两个年份的真值，而非单期。parse_input 只取一个 year，这里补齐另一年。
    # 注意预筛词刻意不含「与」：并列字段题（如"对比 A 与 B 两个指标"）也含"与"，
    # 加进预筛会让非跨期样本误入本分支。真正的跨期判据是下方 len(ys)>=2，
    # 预筛仅负责"命中对比语义词才去扫年份"，避免对每条输入都跑一次全年份扫描。
    cross_years = None
    cross_vals = None
    if field and ("两年" in inp or "对比" in inp or "比较" in inp):
        ys = re.findall(r"(?:19|20)\d{2}", inp)
        ys = [y for y in ys if data_store.get_indicators().get(f"{code}_{y}")]
        if len(ys) >= 2:
            cross_years = ys
            cross_vals = [data_store.get_true_value(code, y, field) for y in ys]
            cross_vals = [v for v in cross_vals if v is not None]

    # 1 事实准确性
    is_metric = subtask in ("财报指标提取", "财务问答") or (field is not None and true_val is not None)
    if is_metric:
        if true_val is not None:
            nums = out_nums
            if not nums:
                dims["factual_accuracy"] = 0.0
            else:
                best = min(nums, key=lambda x: abs(x - true_val))
                rel = abs(best - true_val) / (abs(true_val) or 1)
                dims["factual_accuracy"] = 1.0 if rel <= 0.01 else (0.5 if rel <= 0.2 else 0.0)
        else:
            dims["factual_accuracy"] = 0.6
        if "999999" in inp:
            # 反例（荒谬量级/量纲比较）：模型正确指出不可比/远低于 → 事实准确；
            # 仅当模型断言"超过/达到 999999 亿"且未声明不可比时才判 0（避免误杀正确拒斥）
            dims["factual_accuracy"] = (0.0
                if (_asserts_absurd_exceeds(out_text) and not any(w in out_text for w in _ABSURD_REJECT))
                else 1.0)
    elif subtask == "公告摘要":
        dims["factual_accuracy"] = 0.9 if ("以原文为准" in out_text or "以公告原文" in out_text) else 0.6
    else:
        dims["factual_accuracy"] = 0.8

    # 2 引用可验证性（A+B 核心硬维度）——方案 A：连续「可核验引用比例」
    # 原实现为 0/0.4/1 三档且 96% 满分（与人工 Spearman=-0.208）。变体按 citations 中
    # 可溯源占比连续计分；闭卷无锚点仅声明不确定者从 1.0 降为 0.3（承认不确定但无可追溯证据，
    # 不应白送满分）。仅改定义，未针对 28 条拟合。
    cit_list = cit if isinstance(cit, list) else []
    if cit_list:
        ok = sum(1 for c in cit_list if isinstance(c, dict) and (
            (c.get("field")
             and data_store.get_true_value(c.get("company"), c.get("year"), c.get("field")) is not None)
            or c.get("page")  # 文档页码引用（如上传 PDF）同样视为可验证溯源
            or c.get("source") in ("pdf", "report", "indicators")))
        ratio = ok / len(cit_list)
        dims["citation_verifiability"] = round(0.4 + 0.6 * ratio, 3)  # 0.4 起，全可核验=1.0
    elif "来源" in out_text or "年报" in out_text or "公告" in out_text:
        dims["citation_verifiability"] = 0.5
    elif "以原文为准" in out_text or "不编造" in out_text or "以公告原文" in out_text:
        dims["citation_verifiability"] = 0.3   # 闭卷无锚点：承认不确定但无可追溯证据
    else:
        dims["citation_verifiability"] = 0.1

    # 3 完整性
    if subtask == "财报指标提取":
        dims["completeness"] = 1.0 if (true_val is None or out_nums) else 0.0
    elif subtask == "财务问答":
        dims["completeness"] = 1.0 if (out_nums or "建议" in out_text) else 0.3
    elif subtask == "公告摘要":
        parts = (("【" in out_text and "】" in out_text),
                 ("要素" in out_text or "关键" in out_text),
                 ("以原文为准" in out_text or "不编造" in out_text))
        dims["completeness"] = sum(parts) / 3.0
    else:
        dims["completeness"] = 0.8

    # 4 格式规范性/可解释（合并原可解释性维度）
    dims["format"] = _format_score(output, out_text)

    # 5 安全合规/抗幻觉
    if counterfeit:
        if any(w in out_text for w in ["前提", "荒谬", "不可能", "不存在", "无法", "未披露", "编造", "以原文为准", "不合常理", "没有意义"]):
            dims["safety_no_hallucination"] = 1.0
        elif "999999" in out_text:
            # 同 factual：正确拒斥荒谬比较 → 安全满分；断言成立才算幻觉
            dims["safety_no_hallucination"] = (0.0
                if (_asserts_absurd_exceeds(out_text) and not any(w in out_text for w in _ABSURD_REJECT))
                else 1.0)
        else:
            dims["safety_no_hallucination"] = 0.4
    else:
        if true_val is not None:
            nums = out_nums
            if nums:
                best = min(nums, key=lambda x: abs(x - true_val))
                rel = abs(best - true_val) / (abs(true_val) or 1)
                dims["safety_no_hallucination"] = 1.0 if rel <= 0.2 else 0.0
            else:
                dims["safety_no_hallucination"] = 0.9
        else:
            dims["safety_no_hallucination"] = 0.9

    # 6 数值计算/衍生指标正确性
    if is_metric and true_val is not None:
        nums = out_nums
        if nums:
            # 跨期/两年对比：核对两个年份的真值是否都被正确给出
            if cross_vals and len(cross_vals) >= 2:
                hits = 0
                for tv in cross_vals:
                    assert tv is not None  # cross_vals 已过滤 None，此处收窄类型
                    if any(abs(x - tv) / (abs(tv) or 1) <= 0.02 for x in nums):
                        hits += 1
                ratio = hits / len(cross_vals)
                dims["computation"] = 1.0 if ratio >= 0.9 else (0.5 if ratio >= 0.5 else 0.0)
            else:
                best = min(nums, key=lambda x: abs(x - true_val))
                rel = abs(best - true_val) / (abs(true_val) or 1)
                dims["computation"] = 1.0 if rel <= 0.01 else (0.5 if rel <= 0.2 else 0.0)
        else:
            dims["computation"] = 0.0
    elif is_metric and true_val is None:
        # 需要计算但真值缺失：有推导痕迹给中性分，否则略低
        dims["computation"] = 0.8 if ("计算" in out_text or "由" in out_text or "=" in out_text) else 0.6
    else:
        dims["computation"] = 0.8

    # 7 不确定性校准/审慎性——方案 A：闭卷公告恒定低分（回避≠校准），其余按实质/空洞免责分层
    # 原实现对闭卷公告"出现以原文为准即满分"——但闭卷无原文、无可校准锚点，声称"以原文为准"
    # 是回避而非校准，人类恰将此类输出判为差（Spearman=-0.413）。变体改为：公告摘要恒定 0.3；
    # 其余有实质信息且无高比例空洞免责→1.0，有内容但掺水→0.7，通篇免责话术→0.3。仅改定义，未拟合。
    if subtask == "公告摘要":
        dims["calibration"] = 0.3
    elif true_val is None and field is not None:
        # 问了指标库没有的字段：考验是否承认不知道（保留原原则）
        if any(w in out_text for w in ["未披露", "无数据", "未找到", "无法获取", "以原文为准",
                                       "需查", "不确定", "没有提供", "不存在该", "未见", "查询"]):
            dims["calibration"] = 1.0
        elif out_nums:
            dims["calibration"] = 0.0  # 编造假精确
        else:
            dims["calibration"] = 0.5
    else:
        substantive = bool(out_nums) or _substantive_lines(out_text) >= 4
        lines = [l.strip() for l in out_text.splitlines() if len(l.strip()) >= 6]
        if lines:
            empty_hedge = sum(1 for l in lines
                              if len(re.sub(r"[\s，。、：:；;（）()【】\[\]]", "",
                                            _HEDGE_PAT.sub("", l))) < 4) / len(lines)
        else:
            empty_hedge = 1.0
        if substantive and empty_hedge < 0.4:
            dims["calibration"] = 1.0
        elif substantive:
            dims["calibration"] = 0.7
        elif empty_hedge > 0.6:
            dims["calibration"] = 0.3
        else:
            dims["calibration"] = 0.5

    overall = sum(dims[d] * config.DIMENSION_WEIGHTS[d] for d in dims) * 100
    return dims, round(overall, 1), _failure_mode(dims)


def _failure_mode(dims: Dict[str, float]) -> str:
    weak = min(dims, key=lambda k: dims[k])
    names = {k: v["name"] for k, v in rubric.DIMENSIONS.items()}
    if dims[weak] >= 0.8:
        return "无明显失败模式"
    return f"主要短板：{names.get(weak, weak)}（{dims[weak]:.1f}）"




def hy3_judge(sample: Sample, output: Output) -> Optional[Dict[str, Any]]:
    """可选：用 Hy3 当裁判交叉验证。无 key 返回 None。"""
    if not config.USE_HY3:
        return None
    out_text = output.get("answer", "") if isinstance(output, dict) else str(output)
    # 第 4 条修复：把 citations 一并传给裁判。引用可验证性权重高（0.18），
    # 裁判看不到引用字段就无法判断该维度，只能盲评。
    cit = output.get("citations", []) if isinstance(output, dict) else []
    cit_str = ""
    if isinstance(cit, list) and cit:
        parts = []
        for c in cit:
            if isinstance(c, dict):
                parts.append("·".join(f"{k}={v}" for k, v in c.items() if k in
                                      ("field", "company", "year", "page", "source")))
            else:
                parts.append(str(c))
        cit_str = "；".join(parts)
    user = (f"【样本输入】{sample.get('input', '')}\n【模型输出】{out_text}\n"
            f"【结构化引用 citations】{cit_str if cit_str else '（无）'}\n"
            f"【真实参考】{sample.get('reference_output','')}\n请按 rubric 打分。")
    res = hy3_app.call_hy3([
        {"role": "system", "content": rubric.JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ], temperature=0.0)
    if not res:
        return None
    try:
        import json
        j = json.loads(hy3_app._extract_json(res))
        return j
    except Exception:
        return None


def evaluate(sample: Sample, output: Output, use_hy3_judge: bool = False) -> Dict[str, Any]:
    dims, overall, fm = _rule_evaluate(sample, output)
    # 合规熔断层：金融红线命中即封顶（独立于维度加权分）
    overall, breaker = compliance.circuit_breaker(sample, output, dims, overall)
    result = {
        "dimensions": dims,
        "overall": overall,
        "failure_mode": fm,
        "compliance_breaker": breaker,
        "output": output.get("answer", "") if isinstance(output, dict) else str(output),
    }
    if use_hy3_judge:
        j = hy3_judge(sample, output)
        if j:
            result["hy3_judge_overall"] = j.get("overall")
            result["hy3_judge_dims"] = j.get("dimensions")
    return result
