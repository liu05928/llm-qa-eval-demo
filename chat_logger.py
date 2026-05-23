import json
from datetime import datetime
from json import JSONDecodeError

from config import LOG_PATH, MODEL, USE_MOCK


def load_logs() -> list:
    """
    读取已有日志。

    如果日志文件不存在、为空，或者 JSON 格式损坏，
    就返回一个空列表。
    """

    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return []

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except JSONDecodeError:
        return []


def save_log(question: str, answer: str, mode: str = "general") -> None:
    """
    保存一轮问答日志。

    参数：
        question: 用户问题；
        answer: 模型回答；
        mode: 当前使用的 Prompt 模式。
    """

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logs = load_logs()

    logs.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "mode": mode,
        "model": MODEL,
        "mock": USE_MOCK
    })

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)