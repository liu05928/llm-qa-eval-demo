import json
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
RAG_LOG_FILE = LOG_DIR / "rag_log.json"


def save_rag_log(log_data: dict, log_file: Path = RAG_LOG_FILE):
    """
    保存 RAG 问答日志。

    日志内容包括：
    - 时间
    - 用户问题
    - top_k
    - 模型回答
    - 引用来源
    - 检索到的 chunks
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_file.exists():
        try:
            with log_file.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []

    log_data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logs.append(log_data)

    with log_file.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)