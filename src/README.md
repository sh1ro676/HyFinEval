# A+B 路径实现（金融方向 · 第一版）

基于「犀牛鸟开源-实战任务-混元大语言模型项目」第一个任务：开放式场景的 AI 应用 + 评判标准设计。
本目录为 **路径 A+B（轻量 LLM-as-Judge + 引用可验证）** 的可运行第一版。

## 设计要点
- **应用层（载体）**：输入金融问题 → 检索真实指标（RAG 思路）→ 生成结构化输出 `{answer, citations}`。
- **评估层（评分重心，占 80% 权重）**：
  - 引用可验证（A+B 核心硬维度）：输出须标注来源字段且可追溯至真实数据；
  - 规则校验：数值与 `indicators.json` 真实值比对（事实准确性）；
  - LLM-as-Judge：可切换 Hy3 当裁判交叉验证。
- **评估方法有效性三件套**：判别力 / 一致性 / 对抗性，均已在 `results/` 中验证达标。

## 目录结构
```
a_plus_b/
├── config.py          # 路径、Hy3 API 配置、维度权重（key 走环境变量）
├── data_store.py      # 加载样本与真实指标，按 (代码,年份,字段) 检索
├── baseline_app.py    # 基线应用（无 key 也能跑，确定性抽取）
├── hy3_app.py         # Hy3 应用（RAG 检索 + 生成）+ 通用 Hy3 调用客户端
├── rubric.py          # 5 个评估维度定义与 rubric 分档
├── evaluator.py       # 评估引擎（规则打分 + 可选 Hy3 裁判）
├── run_eval.py        # 完整评测：生成→评估→三件套验证→输出
├── run_demo.py        # 单条演示
├── gen_report.py      # 生成人读分析报告
└── results/           # eval_results.json + eval_report.md
```

## 运行方式
```bash
# 1) 基线版（无需 key，立即看效果）
python run_eval.py
python run_demo.py

# 2) 接入真实 Hy3（需先设置环境变量）
export HY3_API_KEY="你的key"        # 或 HUNYUAN_API_KEY
export HY3_BASE_URL="https://api.hunyuan.cloud.tencent.com/v1"
export HY3_MODEL="hunyuan-turbo"
python run_eval.py --use-hy3        # 应用层与裁判均用 Hy3
python run_eval.py --use-hy3-judge  # 应用用基线，仅裁判用 Hy3 交叉验证
```
> 自部署 Hy3（vLLM/SGLang）只需改 `HY3_BASE_URL`，代码无需改动。
> **红线**：API key 永远走环境变量，绝不硬编码、绝不提交仓库。

## 当前状态（基线版评测结果）
- 应用样本 61 条平均综合分 **81.3**；公告摘要最高（95.5）。
- 判别力达标：好 72 ＞ 中 44.5 ＞ 差 32 ＞ 对抗 32（差/对抗 <50）。
- 一致性（验证集评估分 vs 人工锚定）Spearman = **0.998**。
- 对抗性：对抗样本 32 < 50，通过。
- 详见 `results/eval_report.md`。

## 下一步（待你确认/提供 key）
1. 提供 Hy3 key 后切换 `--use-hy3`，应用输出将带结构化 citations，各档梯度更清晰。
2. 如需冲分，可在应用层补 RAG 全文检索（年报 PDF 受反爬限制，目前用结构化指标作上下文）。
3. 补充 demo 视频/GIF（展示 应用→评估 完整流程）。
