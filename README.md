# HyFinEval — 混元金融评估应用与评判标准

> **犀牛鸟开源实战任务 · 混元大语言模型项目（个人作品）**
>
> 本项目为个人参赛作品，非官方出品，仅供学习与评测参考。任务方向：开放式场景下的 **AI 应用 + 评判标准设计**，模型能力基于腾讯混元（Hy3 / HunYuan）大语言模型 API 构建。

## 项目简介

本项目以**金融分析**为落地场景，构建了一个基于混元大模型的金融问答 / 指标提取应用，并**重点设计了可复现、可验证的评估方法**（评分重心占任务的 80%）。评估方法包含 5 个量化维度与一套「引用可验证性」硬规则，并通过「判别力 / 一致性 / 对抗性」三件套实验验证其有效性。

## 仓库结构

```
HyFinEval/
├── README.md                  # 本文件（项目总览）
├── LICENSE                    # MIT（个人作品，可自由参考）
├── .gitignore
├── .env.example               # API key 模板（不提交真实 key）
├── build_finance_samples.py   # 零成本样本集构建脚本（akshare + 巨潮）
├── samples.json / samples.csv # 评测样本集（81 条，含难例/反例）
├── data_cache/                # 真实财务数据缓存（indicators.json 等）
├── src/                       # A+B 路径实现：应用层 + 评估层
│   ├── config.py              # 路径 / API / 维度权重 配置
│   ├── data_store.py          # 样本与真实指标检索
│   ├── baseline_app.py        # 基线应用（无 key 可跑）
│   ├── hy3_app.py             # 混元 Hy3 应用生成（需 key）
│   ├── rubric.py              # 5 维度 rubric 定义
│   ├── evaluator.py           # 评估引擎（引用可验证规则 + 可选 LLM-judge）
│   ├── run_eval.py            # 完整评测 + 三件套验证
│   ├── run_demo.py            # 单条演示
│   ├── gen_report.py          # 生成人读分析报告
│   └── results/               # eval_results.json / eval_report.md
└── docs/                      # 提交考官的方案 MD（考生口吻）
    ├── 路径A-轻量LLM-as-Judge.md
    ├── 路径A+B-RAG引用可验证.md
    └── 路径C-多Agent对抗验证.md
```

## 快速开始

### 1. 安装依赖
```bash
pip install akshare pandas requests openai
```

### 2. 运行基线评测（无需 API key）
```bash
python src/run_eval.py
```
使用确定性基线应用 + 规则评估器，立即产出 `src/results/eval_results.json` 与 `eval_report.md`。

### 3. 接入混元 Hy3（可选，效果更佳）
复制 `.env.example` 为 `.env` 并填入你的 key（**绝不提交 `.env`**）：
```powershell
$env:HY3_API_KEY="你的key"
```
然后运行：
```bash
python src/run_eval.py --use-hy3          # 应用用真实模型生成（带 citations）
python src/run_eval.py --use-hy3-judge   # 评估器用 Hy3 作交叉裁判
```

## 评估方法（核心）

5 个量化维度，权重 `事实准确性 0.25 / 引用可验证性 0.25 / 完整性 0.15 / 格式 0.10 / 安全合规·抗幻觉 0.25`，对照 `data_cache/indicators.json` 真实财务数据规则打分后加权合成 0–100 总分。评估有效性通过三件套验证：

- **判别力**：验证集「好 > 中 > 差 > 对抗」严格递减
- **一致性**：规则分 vs 人工锚定评级 Spearman 相关（实测 0.998）
- **对抗性**：对抗样本（堆术语 / 伪引用 / 编造）得分 < 50，证明评估器可识破

详细评分标准见 `src/rubric.py` 与 `src/evaluator.py`。

## 样本集说明（零成本构建）

- 数据源：akshare 财务分析指标（真实绝对值+比率）、巨潮 cninfo 披露列表（真实公告标题）
- 成本：全部公开、无需鉴权、零成本；不含任何 API key
- 总量：81 条（应用样本 61 + 验证集 20）；应用样本中难例+反例占 41%，满足任务书 ≥30% 要求
- 反例：基于真实数据**确定性改错**（数量级错误、错误前提、编造内容），不引入幻觉数据
- 字段：`id, subtask, difficulty, input, reference_output, human_anchor_rating, source, is_counterfeit, notes`
- 复现：`python build_finance_samples.py`（需联网拉取真实数据）

## 安全与合规

- API key 仅通过环境变量 / `.env` 传入，代码不硬编码、不提交仓库
- 所有样本为公开数据或基于公开数据的确定性改写，无隐私 / 商业秘密
- 金融内容仅用于算法评测，不构成任何投资建议

---

参赛选手：\_\_\_\_\_\_\_\_\_\_　|　方向：金融分析 — AI 应用与评判标准设计
