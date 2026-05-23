import os
from pathlib import Path

from dotenv import load_dotenv


# 当前项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 读取项目根目录下的 .env 文件
load_dotenv(BASE_DIR / ".env")

# DeepSeek API Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek API 请求地址
BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com/chat/completions"
)

# 模型名称
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 是否开启模拟模式
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# 日志文件路径
LOG_PATH = BASE_DIR / "logs" / "chat_log.json"

# 测试问题集路径
TEST_QUESTIONS_PATH = BASE_DIR / "data" / "test_questions.json"

# 评测结果路径
EVAL_RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"