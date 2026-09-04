"""方案 A：释放被浪费的方差（修维度定义，不动标注、不改权重）。

背景：原始 rubric 三维度在 28 条真人盲标上接近零方差或显著负相关：
  - format             28/28 = 1.0（零方差，0 判别力）
  - citation_verifiability  96.4% 满分（Spearman vs 人工 = -0.208）
  - calibration              Spearman vs 人工 = -0.413（话术越多分越高，反向）

本模块对这三个维度做 **定义层面** 的修正（均有独立理由，非针对 28 条拟合）：
  A1 format   : 结构覆盖度 × 信息密度 × (1 - 冗余率 - 纯免责行占比)
  A2 citation : 连续「可核验引用比例」；闭卷无锚点降至 0.3（不再因"以原文为准"白送满分）
  A3 calibration: 重定义为「有实质信息且不靠空洞免责填充」；通篇话术从满分改为低分

用法：variant_rule_evaluate(sample, output) 返回与 evaluator._rule_evaluate 同构的
(dims, overall, failure_mode)，仅三维换成本模块的变体。消融脚本 ablation_dims.py 负责
在 28 条上对比「原维度 ρ」与「变体维度 ρ」，有效后再并入 evaluator.py。
"""

import re

import evaluator
import config


# ---------- A1 format：回退原版（惰性零方差）----------
# 实测 format 变体（信息密度×冗余×纯免责行占比）与人类档位呈反向（ρ=-0.353）：
# 人类"优"档平均 0.857 反而低于"中"0.888/"差"0.883——密度/冗余判据惩罚的恰是人类
# 喜欢的输出。证明在本任务中"格式密度"与内容质量不相关，强加方差只会帮倒忙。
# 故方案 A 对 format 不做定义变更（保留原零方差惰性维度），把权重预算留给 citation/calibration。

def _format_variant(out_text, output):
    return evaluator._format_score(output, out_text)


# ---------- A2 citation 变体 ----------

def _citation_variant(subtask, out_text, cit):
    """连续「可核验引用比例」。

    原实现：公告摘要出现"以原文为准"即 1.0；有 citations 列表即 1.0/0.4。
    问题：闭卷公告既无原文也无锚点，却因一句声明拿满分；且 0/0.4/1 三档零方差。
    变体：
      - 有结构化 citations：按可溯源占比连续计分 0.4~1.0
      - 文本提及来源但无锚点：0.5
      - 闭卷无引用、仅声明不确定：0.3（承认不确定但无可追溯证据）
      - 既无引用又无声明：0.1
    """
    out_text = str(out_text)
    if cit and isinstance(cit, list) and len(cit) > 0:
        ok = sum(1 for c in cit if isinstance(c, dict) and (
            (c.get("field")
             and evaluator.data_store.get_true_value(c.get("company"), c.get("year"), c.get("field")) is not None)
            or c.get("page")
            or c.get("source") in ("pdf", "report", "indicators")))
        ratio = ok / len(cit)
        return round(0.4 + 0.6 * ratio, 3)  # 0.4 起，全可核验=1.0

    if "来源" in out_text or "年报" in out_text or "公告" in out_text:
        return 0.5
    if "以原文为准" in out_text or "不编造" in out_text or "以公告原文" in out_text:
        return 0.3
    return 0.1


# ---------- A3 calibration 变体（v2）----------

def _calibration_variant(out_text, subtask, true_val, field):
    """重定义为「恰当的审慎性」而非「免责话术奖励」。

    根因：原实现对闭卷公告"出现以原文为准即满分"——但闭卷无原文、无可校准锚点，
    声称"以原文为准"是**回避**而非校准，人类恰将此类输出判为差（ρ=-0.413）。
    变体：
      - 公告摘要（闭卷、无锚点）：恒定 0.3 —— 无法对不存在的数据做恰当校准，低分且诚实。
      - 其余：有实质信息且无高比例空洞免责 → 1.0；有内容但掺水 → 0.7；
              通篇免责话术（无实质+纯免责行占比高）→ 0.3；其余 → 0.5。
    """
    out_text = str(out_text)
    if subtask == "公告摘要":
        return 0.3

    substantive = bool(evaluator._extract_numbers(out_text)) or evaluator._substantive_lines(out_text) >= 4
    lines = [l.strip() for l in out_text.splitlines() if len(l.strip()) >= 6]
    if lines:
        empty_hedge = sum(1 for l in lines
                          if len(re.sub(r"[\s，。、：:；;（）()【】\[\]]", "",
                                        evaluator._HEDGE_PAT.sub("", l))) < 4) / len(lines)
    else:
        empty_hedge = 1.0

    if substantive and empty_hedge < 0.4:
        return 1.0
    if substantive:
        return 0.7   # 有内容但掺水
    if empty_hedge > 0.6:
        return 0.3   # 通篇免责话术：假审慎，最差校准
    return 0.5


# ---------- 组合：换三维后重算 ----------

def variant_rule_evaluate(sample, output):
    """复用 evaluator._rule_evaluate 的结构，仅替换 format/citation/calibration 三维。

    返回 (dims, overall, failure_mode)，其余四维（factual/completeness/safety/computation）
    与原实现完全一致，确保消融只测 A1~A3 的影响。
    """
    dims, overall, fm = evaluator._rule_evaluate(sample, output)
    out_text = output.get("answer", "") if isinstance(output, dict) else str(output)
    cit = output.get("citations", []) if isinstance(output, dict) else []
    code, year, field = evaluator.data_store.parse_input(sample["input"])
    true_val = evaluator.data_store.get_true_value(code, year, field) if field else None

    dims["format"] = _format_variant(out_text, output)
    dims["citation_verifiability"] = _citation_variant(sample["subtask"], out_text, cit)
    dims["calibration"] = _calibration_variant(out_text, sample["subtask"], true_val, field)

    overall = sum(dims[d] * config.DIMENSION_WEIGHTS[d] for d in dims) * 100
    return dims, round(overall, 1), evaluator._failure_mode(dims)
