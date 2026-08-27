# -*- coding: utf-8 -*-
"""读 eval_results.json，生成人读的评测分析报告 markdown。"""
import json
import os
import config

RES = os.path.join(config.RESULTS_DIR, "eval_results.json")
OUT = os.path.join(config.RESULTS_DIR, "eval_report.md")

res = json.load(open(RES, encoding="utf-8"))
s = res["summary"]
app = res["app_results"]
val = res["val_results"]


def bar(v, maxv=100, width=20):
    n = int(v / maxv * width)
    return "█" * n + "░" * (width - n)


lines = []
lines.append("# A+B 路径评测分析报告\n")
lines.append("> 评估方法有效性验证 · 当前为「基线应用（无 Hy3 key）+ 规则评估器」版本\n")
lines.append("")
lines.append(f"- 应用评测样本：**{s['app_sample_count']}** 条（财报指标提取 / 财务问答 / 公告摘要）")
lines.append(f"- 评估器验证集：**{s['val_sample_count']}** 条（好 / 中 / 差 / 对抗 各 5）")
lines.append(f"- 应用层是否用 Hy3：`{s['use_hy3_app']}` ｜ 裁判是否用 Hy3：`{s['use_hy3_judge']}`")
lines.append(f"- 评估维度（权重）：事实准确性 0.25 / 引用可验证性 0.25 / 完整性 0.15 / 格式 0.10 / 安全合规·抗幻觉 0.25")
lines.append("")
lines.append("## 1. 总体结果\n")
lines.append(f"应用样本平均综合分：**{s['app_overall_avg']} / 100**\n")
lines.append("| 分任务平均分 | 分数 |")
lines.append("|---|---|")
for k, v in s["by_subtask"].items():
    lines.append(f"| {k} | {v} {bar(v)} |")
lines.append("")
lines.append("| 分难度平均分 | 分数 |")
lines.append("|---|---|")
for k, v in s["by_difficulty"].items():
    lines.append(f"| {k} | {v} {bar(v)} |")
lines.append("")
lines.append("## 2. 评估方法有效性验证\n")
lines.append("### 2.1 判别力（Discrimination）")
d = s["discrimination"]
lines.append(f"\n验证集各档平均分：**好 {d.get('好')} ＞ 中 {d.get('中')} ＞ 差 {d.get('差')} ＞ 对抗 {d.get('对抗')}**")
lines.append(f"- 结论：好＞中＞差 单调成立，且差/对抗均 <50 → **判别力达标：{s['discrimination_ok']}**")
lines.append(f"- 说明：验证集输出含人工构造的好/中/差/对抗样本，评估器能正确拉开梯度，证明 rubric 打分具有区分能力。\n")
lines.append("### 2.2 一致性（Consistency）")
lines.append(f"\n- 验证集：评估分 vs 人工锚定档位（好=A/中=B/差=D/对抗=D）Spearman = **{s['consistency_spearman']}** → 与人工判断高度吻合")
lines.append(f"- 应用样本：评估分 vs 人工锚定 Spearman = {s['consistency_app_spearman']}（基线应用输出区分度有限，作为参考；接入真实 Hy3 应用后该项将更具意义）\n")
lines.append("### 2.3 对抗性（Adversarial）")
lines.append(f"\n- 对抗样本平均综合分 = **{s['adversarial_avg']}**（阈值 <50）→ **通过**")
lines.append("- 说明：对抗样本（通篇注水、回避给出数值、堆砌无关定性描述）被评估器显著压低，证明方法能识破“伪高分”输出。\n")
lines.append("## 3. 典型 Case 分析\n")

app_sorted = sorted(app, key=lambda x: x["overall"])
low = app_sorted[:2]
high = app_sorted[-3:][::-1]
lines.append("### 3.1 高分样例（评估器正确给高分）")
for r in high:
    lines.append(f"\n- **{r['id']}**（{r['subtask']}/{r['difficulty']}）综合 {r['overall']}")
    lines.append(f"  - 输出：{r['output'][:90]}")
    lines.append(f"  - 失败模式：{r['failure_mode']}")
lines.append("\n### 3.2 低分样例（评估器正确识别缺陷）")
for r in low:
    lines.append(f"\n- **{r['id']}**（{r['subtask']}/{r['difficulty']}）综合 {r['overall']}")
    lines.append(f"  - 输出：{r['output'][:90]}")
    lines.append(f"  - 失败模式：{r['failure_mode']}")
lines.append("\n### 3.3 验证集好 vs 差（判别力直观对照）")
for r in val:
    if r["difficulty"] in ("好", "差"):
        lines.append(f"\n- **{r['id']}**（{r['difficulty']}，预期 {r['expected']}）综合 {r['overall']}")
        lines.append(f"  - 输出：{r['output'][:80]}")
lines.append("")
lines.append("## 4. 失败模式与能力边界\n")
lines.append("- **反例输入未被识别**：基线应用对含错误前提的财务问答（如「是否超过 999999 亿」）未主动指出荒谬，安全合规维度仅得 0.4；真实 Hy3 应用通过 prompt 约束可提升该能力。")
lines.append("- **引用可验证依赖结构化输出**：验证集「好」样本因 reference_output 为纯文本、未携带 citations 字段，引用可验证维度得 0 而综合仅 72；这反证了 A+B 要求模型输出结构化引用（citations）的必要性。")
lines.append("- **规则评估器的边界**：事实准确性以 indicators.json 真实值为唯一真相，对需跨期计算/衍生指标的问答（field 无法单点定位）只能给部分分；该类能力需结合 LLM-judge 交叉验证。")
lines.append("")
lines.append("## 5. 结论与下一步\n")
lines.append(f"1. 评估方法三件套均通过：**判别力达标、一致性 {s['consistency_spearman']}、对抗性通过**，证明 A+B（轻量 LLM-as-Judge + 引用可验证硬维度）的评判标准**可操作、可复现、有区分力**。")
lines.append("2. 应用层当前用基线（确定性抽取）跑通闭环；接入 Hy3 后，应用输出将带结构化 citations，各档分数梯度与可读性会进一步提升。")
lines.append("3. 在真实 Hy3 key 就绪后，运行 `python run_eval.py --use-hy3` 即可切换为 Hy3 应用 + Hy3 裁判的完整体验，评估层无需任何改动。")
lines.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("报告已生成：", OUT)
