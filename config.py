import os
from pathlib import Path

from dotenv import load_dotenv


# 当前项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 读取项目根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")


# =========================
# 大模型 API 配置
# =========================

# 硅基流动 API Key
# 注意：这里虽然变量名叫 DEEPSEEK_API_KEY，但实际可以填写硅基流动的 API Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 硅基流动 OpenAI 兼容接口地址
BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.siliconflow.cn/v1/chat/completions"
)

# 硅基流动平台上的模型名称
# 常见示例：
# deepseek-ai/DeepSeek-V3
# deepseek-ai/DeepSeek-R1
MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-ai/DeepSeek-V3"
)

# 为了兼容 llm_client.py 中可能使用 MODEL_NAME 的写法
MODEL_NAME = MODEL

# 是否开启模拟模式
# USE_MOCK=true  使用 Mock 回答
# USE_MOCK=false 调用真实 API
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"


# =========================
# 本地文件路径配置
# =========================

# 日志文件路径
LOG_PATH = BASE_DIR / "logs" / "chat_log.json"

# 测试问题集路径
TEST_QUESTIONS_PATH = BASE_DIR / "data" / "test_questions.json"

# 评测结果路径
EVAL_RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"