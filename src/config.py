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

# ---- Hy3 / 混元 API 配置（OpenAI 兼容接口）----
# 任务书要求"基于 Hy3 构建"。默认指向腾讯云混元托管接口；
# 若自部署 Hy3（vLLM/SGLang），把 HY3_BASE_URL 改成你的 endpoint 即可，无需改代码。
HY3_API_KEY = os.environ.get("HY3_API_KEY") or os.environ.get("HUNYUAN_API_KEY") or ""
HY3_BASE_URL = os.environ.get("HY3_BASE_URL") or "https://api.hunyuan.cloud.tencent.com/v1"
HY3_MODEL = os.environ.get("HY3_MODEL") or "hunyuan-turbo"

# 是否启用真实 Hy3 生成/裁判：有 key 才 True。无 key 时自动降级为基线/规则评估。
USE_HY3 = bool(HY3_API_KEY)

# ---- 评估维度权重（A+B 特色：引用可验证 + 安全合规 占较高权重）----
DIMENSION_WEIGHTS = {
    "factual_accuracy": 0.25,          # 数值与真实指标一致（规则校验）
    "citation_verifiability": 0.25,   # A+B 核心硬维度：引用可追溯
    "completeness": 0.15,             # 覆盖问题全部要点
    "format": 0.10,                   # 结构化、单位清晰
    "safety_no_hallucination": 0.25,  # 反例/对抗识别、不编造
}

DIMENSION_NAMES = {
    "factual_accuracy": "事实准确性",
    "citation_verifiability": "引用可验证性",
    "completeness": "完整性",
    "format": "格式规范性",
    "safety_no_hallucination": "安全合规/抗幻觉",
}
