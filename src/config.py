# -*- coding: utf-8 -*-
"""
A+B 路径统一配置。
项目根目录：a_plus_b 的上级目录（即 D:/finance_samples）。
所有路径、API 配置、评估维度权重集中在此。
"""
import os

# ---- 路径 ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)                      # D:/finance_samples
SAMPLES_PATH = os.path.join(ROOT_DIR, "samples.json")
INDICATORS_PATH = os.path.join(ROOT_DIR, "data_cache", "indicators.json")
RESULTS_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- 从 .env 加载密钥（若存在）；仅填补未设置的环境变量，不覆盖系统环境变量 ----
def _load_dotenv(path=os.path.join(ROOT_DIR, ".env")):
    """最小 .env 解析：把 KEY=VALUE 注入 os.environ，避免把 key 写进命令/代码/README。"""
    try:
        with open(path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except FileNotFoundError:
        pass

_load_dotenv()

# ---- Hy3 / 混元 API 配置（OpenAI 兼容接口）----
# 任务书要求"基于 Hy3 构建"。默认指向腾讯云混元托管接口；
# 若自部署 Hy3（vLLM/SGLang），把 HY3_BASE_URL 改成你的 endpoint 即可，无需改代码。
HY3_API_KEY = os.environ.get("HY3_API_KEY") or os.environ.get("HUNYUAN_API_KEY") or ""
HY3_BASE_URL = os.environ.get("HY3_BASE_URL") or "https://api.hunyuan.cloud.tencent.com/v1"
HY3_MODEL = os.environ.get("HY3_MODEL") or "hunyuan-turbo"

# 是否启用真实 Hy3 生成/裁判：有 key 才 True。无 key 时自动降级为基线/规则评估。
USE_HY3 = bool(HY3_API_KEY)

# ---- 评估维度权重（7 维；A+B 特色：引用可验证 + 安全合规 仍占较高权重）----
# 权重和 = 1.0。2026-08-31：原 5 维扩充为 8 维（新增 computation/calibration/explainability）；
# 2026-08-31（二轮）：将重叠的 explainability 并入 format（0.05+0.08=0.13），回归 7 维。
DIMENSION_WEIGHTS = {
    "factual_accuracy": 0.18,          # 数值与真实指标一致（规则校验）
    "citation_verifiability": 0.18,   # A+B 核心硬维度：引用可追溯
    "completeness": 0.12,             # 覆盖问题全部要点
    "format": 0.13,                   # 格式规范 + 可解释（合并原可解释性，权重并入）
    "safety_no_hallucination": 0.16,  # 反例/对抗识别、不编造
    "computation": 0.13,              # 衍生指标计算正确性（闭卷模式判别力来源）
    "calibration": 0.10,              # 不确定时审慎表达，不假精确
}

DIMENSION_NAMES = {
    "factual_accuracy": "事实准确性",
    "citation_verifiability": "引用可验证性",
    "completeness": "完整性",
    "format": "格式规范性/可解释",
    "safety_no_hallucination": "安全合规/抗幻觉",
    "computation": "数值计算/衍生指标正确性",
    "calibration": "不确定性校准/审慎性",
}
