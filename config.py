import os
from pathlib import Path

from dotenv import load_dotenv


# 当前项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 读取项目根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")


# =========================
# 运行模式配置
# =========================

# USE_MOCK=true  使用 Mock 回答，不调用真实 API
# USE_MOCK=false 调用硅基流动真实 API
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

# 生成模型后端：
# mock      本地模拟回答，用于离线演示和单元校验
# api       SiliconFlow / DeepSeek API，作为 baseline 或 fallback
# local_sft OpenAI-compatible 本地/云端微调模型服务
GENERATION_BACKEND = os.getenv(
    "GENERATION_BACKEND",
    "mock" if USE_MOCK else "api",
)


# =========================
# 硅基流动统一 API 配置
# =========================

# 新版统一变量名
SILICONFLOW_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    os.getenv("DEEPSEEK_API_KEY", "")
)

SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1"
)

# 生成式大模型
CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3")
)

# 本地或云端 SFT 模型服务，后续云端训练完成后只需要把 endpoint 填到 .env。
LOCAL_SFT_MODEL = os.getenv(
    "LOCAL_SFT_MODEL",
    "qwen2.5-3b-instruct-edu-lora",
)

LOCAL_SFT_BASE_URL = os.getenv(
    "LOCAL_SFT_BASE_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)

LOCAL_SFT_API_KEY = os.getenv("LOCAL_SFT_API_KEY", "EMPTY")


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


LOCAL_SFT_TEMPERATURE = _get_float_env("LOCAL_SFT_TEMPERATURE", 0.2)
LOCAL_SFT_TOP_P = _get_float_env("LOCAL_SFT_TOP_P", 0.8)
LOCAL_SFT_MAX_TOKENS = _get_int_env("LOCAL_SFT_MAX_TOKENS", 512)
LOCAL_SFT_TIMEOUT_SECONDS = _get_int_env("LOCAL_SFT_TIMEOUT_SECONDS", 120)
LOCAL_SFT_REPETITION_PENALTY = _get_float_env("LOCAL_SFT_REPETITION_PENALTY", 1.1)

# Embedding 向量模型
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3"
)

# Rerank 重排序模型
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "BAAI/bge-reranker-v2-m3"
)


# =========================
# 兼容旧代码变量
# =========================
# 旧代码里可能还在使用 API_KEY、BASE_URL、MODEL、MODEL_NAME。
# 这里保留这些变量，避免 app.py、llm_client.py、evaluator.py 报错。

API_KEY = SILICONFLOW_API_KEY

# 注意：旧版 llm_client.py 可能直接把 BASE_URL 当作 chat completions 完整地址使用，
# 所以这里保留完整的 /chat/completions。
BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    f"{SILICONFLOW_BASE_URL}/chat/completions"
)

MODEL = CHAT_MODEL
MODEL_NAME = CHAT_MODEL


# =========================
# 本地文件路径配置
# =========================

# 日志文件路径
LOG_PATH = BASE_DIR / "logs" / "chat_log.json"

# 测试问题集路径
TEST_QUESTIONS_PATH = BASE_DIR / "data" / "test_questions.json"

# 评测结果路径
EVAL_RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"
