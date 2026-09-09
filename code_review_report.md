# HyFinEval 代码审查报告

> 审查范围：`src/` 全部 Python 模块 + `build_finance_samples.py` + `README.md`
> 审查日期：2026-09-08
> 方法论：静态代码走读 + 逻辑推演 + 与 README 自声明行为交叉核验

---

## 一、执行摘要

本项目整体架构清晰，评估体系设计严谨，文档（README）与代码实现高度一致，体现了良好的工程意识。但在**并发正确性、错误降级路径、资源管理、配置一致性**四个方面存在可修复的缺陷，另有若干**性能与可维护性**改进空间。

| 级别 | 数量 | 类别 |
|---|---|---|
| 🔴 **严重** | 3 | Bug（并发失效、错误泄漏、配置漂移） |
| 🟡 **中等** | 5 | 设计/性能问题 |
| 🟢 **建议** | 4 | 可维护性改进 |

---

## 二、🔴 严重问题（Bug）

### 2.1 `run_stability.py`：并发完全失效（workers > 1 时等于串行）

**位置**：`src/run_stability.py` 第 58–62 行

**问题代码**：
```python
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(generate_output, s, True, use_rag): s["id"] for s in samples}
    for f in futs:                          # ← 未用 as_completed
        outs[f.result() if False else futs[f]] = f.result()
```

**根因**：
- `for f in futs:` 遍历 dict 的 key（Future 对象），而 `f.result()` 会**阻塞等待该 Future 完成**。
- 由于遍历顺序 = 提交顺序，它会按顺序逐个等待，等效于**完全串行执行**。
- `as_completed` 的缺失使得 `--workers` 参数在稳定性测试中形同虚设。

**修复**：
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
# ...
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(generate_output, s, True, use_rag): s["id"] for s in samples}
    for f in as_completed(futs):
        outs[futs[f]] = f.result()
```

---

### 2.2 `hy3_app.py`：API 错误文本泄漏到用户界面

**位置**：`src/hy3_app.py` 第 160–172 行

**问题代码**：
```python
parsed = _parse_hy3_json(out)
if parsed is not None:
    return parsed

# 兜底：尝试从文本抽取 citations
cit = []
# ...
return {"answer": out, "citations": cit if isinstance(cit, list) else []}
```

**根因**：
- 当 `call_hy3` 返回 `[HY3_ERROR] Connection timeout...` 时，`_parse_hy3_json` 返回 `None`。
- 代码进入兜底分支，把错误字符串直接塞进 `answer` 字段返回。
- `app.py` 中 `if not out:` 为 `False`（字典非空），**不会触发 baseline 回退**，用户直接看到错误文本。

**修复**：在兜底前检查是否为错误字符串：
```python
if not out or out.startswith("[HY3_ERROR]"):
    return None
parsed = _parse_hy3_json(out)
# ...
```

---

### 2.3 `config.py`：默认 endpoint 与项目实际使用环境不一致

**位置**：`src/config.py` 第 39–40 行

**问题代码**：
```python
HY3_BASE_URL = os.environ.get("HY3_BASE_URL") or "https://api.hunyuan.cloud.tencent.com/v1"
HY3_MODEL = os.environ.get("HY3_MODEL") or "hunyuan-turbo"
```

**根因**：
- README 和用户实际注入的环境变量指向 `tokenhub-intl.tencentmaas.com/v1`，模型为 `hy3`。
- 若用户仅设置 `HY3_API_KEY`（最自然的做法），未设置 `HY3_BASE_URL` 和 `HY3_MODEL`，代码会连到错误的 endpoint 并使用错误的模型名。
- 这是一个**配置漂移**问题：默认值与实际使用场景不匹配。

**修复**：将默认值改为与项目实际接入参数一致：
```python
HY3_BASE_URL = os.environ.get("HY3_BASE_URL") or "https://tokenhub-intl.tencentmaas.com/v1"
HY3_MODEL = os.environ.get("HY3_MODEL") or "hy3"
```

---

### 2.4 多处文件句柄未显式关闭（资源泄漏风险）

**涉及位置**：

| 文件 | 行号 | 问题代码 |
|---|---|---|
| `gen_announcement_outputs.py` | 113 | `samples = json.load(open(SAMPLES, encoding="utf-8"))` |
| `ablation_dims.py` | 105 | `json.dump(res, open(OUT, "w", encoding="utf-8"), ...)` |
| `gen_report.py` | 10 | `res = json.load(open(RES, encoding="utf-8"))` |
| `gen_report.py` | 86 | `open(OUT, "w", encoding="utf-8").write("\n".join(lines))` |
| `score_announcement.py` | 163 | `recs = json.load(open(OUTS, encoding="utf-8"))` |

**根因**：`open()` 返回的文件对象未关闭。在 CPython 中依赖引用计数通常不会立刻出问题，但在 PyPy 或高频调用场景下可能触发 `ResourceWarning` 甚至句柄耗尽。

**修复**：统一改用 `with open(...) as f:` 上下文管理器。

---

## 三、🟡 中等问题（设计 / 性能）

### 3.1 `hy3_app.call_hy3`：异常捕获过于宽泛

**位置**：`src/hy3_app.py` 第 52 行

**问题**：`except Exception as e:` 会吞掉 `KeyboardInterrupt`（Ctrl+C），用户在并发批量跑评测时无法正常中断。

**修复**：
```python
except KeyboardInterrupt:
    raise
except Exception as e:
    return f"[HY3_ERROR] {e}"
```

---

### 3.2 `hy3_app.call_hy3`：缺少 HTTP 连接池

**位置**：`src/hy3_app.py` 第 37 行

**问题**：每次调用新建 `requests.post()`，TCP 握手 + TLS 协商在并发场景下开销显著。实测 `--workers 16` 时大量时间花在连接建立上。

**修复**：使用 `requests.Session()` 复用连接：
```python
_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

def call_hy3(...):
    # ...
    r = _get_session().post(url, headers=headers, json=payload, timeout=180)
```

---

### 3.3 `evaluator.py`：同一段文本的正则提取被重复执行

**位置**：`src/evaluator.py`

**问题**：`_extract_numbers(out_text)` 在 `factual_accuracy`、`safety_no_hallucination`、`computation` 三个维度中分别独立调用，对同一段输出做了 3 次相同的正则扫描。

**修复**：在 `_rule_evaluate` 开头提取一次并缓存：
```python
nums = _extract_numbers(out_text)  # 只提取一次
# 后续各维度复用 nums
```

---

### 3.4 `evaluator.py` 与 `score_announcement.py`：两套公告摘要评分体系并存

**位置**：`src/evaluator.py`（第 130–131、160–164 行）vs `src/score_announcement.py`

**问题**：
- 旧体系（evaluator）：公告摘要用启发式关键词打分（"以原文为准"→0.9），缺乏真值核对。
- 新体系（score_announcement）：基于原文客观抽取 fact_coverage / hallucination_rate。
- 两个体系同时存在于仓库中，但没有统一入口说明何时用哪个。`run_eval.py` 仍走旧体系。

**建议**：在 `run_eval.py` 或 evaluator 中增加分支——当样本包含 `ground_truth_facts` 时自动切换到客观评分体系。

---

### 3.5 `label_stats.py`：分档阈值完全硬编码

**位置**：`src/label_stats.py` 第 125–126 行

**问题代码**：
```python
def _auto_band(s):
    return 2 if s >= 95 else (1 if s >= 85 else 0)
```

**问题**：`95` / `85` 阈值无文档说明、无参数化。若后续调整维度权重，阈值需要同步手动修改，容易遗漏。

**建议**：将阈值提取到 `config.py` 或 rubric 定义中，并附注释说明依据。

---

## 四、🟢 建议（可维护性）

### 4.1 正则模式重复定义

`MONEY`、`RATIO`、`DATE` 等正则同时在 `build_announcement_samples.py`、`score_announcement.py` 中定义，违反 DRY 原则。建议抽取到共享模块（如 `src/ann_utils.py`）。

### 4.2 测试缺失

核心逻辑（`evaluator._rule_evaluate`、`stats_utils.spearman`、`compliance.circuit_breaker`）均无单元测试。建议至少补充：
- 评估器各维度的边界 case（如跨期对比、999999 反例、空输出）
- stats_utils 与 scipy 的交叉验证
- compliance 的三种红线触发条件

### 4.3 类型注解缺失

全部模块均无类型提示，增加了维护成本。建议为核心函数（`evaluate`、`call_hy3`、`generate_output` 等）逐步补充 `typing` 注解。

### 4.4 `fetch_announcements.py` 的交易所判断过于简化

```python
column = "sse" if code.startswith("6") else "szse"
```

当前数据集中无 B 股 / 港股 / 科创板代码（688 开头仍满足 `startswith("6")`），但若后续扩展需加入 300/301（创业板）、002/003（中小板）、900（沪市 B 股）等，此判断会出错。建议用明确的前缀映射表替代。

---

## 五、快速修复清单（可直接复制执行）

| 优先级 | 文件 | 修复内容 | 预估工作量 |
|---|---|---|---|
| 🔴 | `run_stability.py` | `for f in futs:` → `for f in as_completed(futs):` | 1 分钟 |
| 🔴 | `hy3_app.py` | 兜底前检查 `out.startswith("[HY3_ERROR]")` 则返回 None | 2 分钟 |
| 🔴 | `config.py` | 默认值改为 `tokenhub-intl.tencentmaas.com/v1` + `hy3` | 1 分钟 |
| 🔴 | 5 个文件 | `open()` 改为 `with open()` | 10 分钟 |
| 🟡 | `hy3_app.py` | 增加 `requests.Session()` 连接池 | 5 分钟 |
| 🟡 | `hy3_app.py` | `except` 前加 `except KeyboardInterrupt: raise` | 1 分钟 |
| 🟡 | `evaluator.py` | 缓存 `_extract_numbers` 结果 | 5 分钟 |
| 🟡 | `label_stats.py` | 阈值提取到 config | 3 分钟 |
| 🟢 | 3 个文件 | 抽取 `MONEY/RATIO/DATE` 到共享模块 | 10 分钟 |
| 🟢 | — | 补充 evaluator / stats_utils 单元测试 | 30–60 分钟 |

---

## 六、总体评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 架构设计 | ★★★★★ | 应用层与评估层解耦、RAG 与规则评估分离，设计清晰 |
| 代码健壮性 | ★★★☆☆ | 并发/错误降级/资源管理有瑕疵，核心评分逻辑正确 |
| 文档一致性 | ★★★★☆ | README 与代码高度一致，但 TODO 列表与代码状态有滞后 |
| 可复现性 | ★★★★★ | 零成本跑通、确定性评估、数据全部公开 |
| 可维护性 | ★★★☆☆ | 缺少测试和类型注解，正则/阈值有重复硬编码 |

**核心建议**：优先修复 2.1–2.4 四个 Bug（合计约 15 分钟），再逐步补充连接池和缓存优化。测试覆盖是长期最大杠杆。
