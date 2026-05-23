import json
from datetime import datetime
from json import JSONDecodeError
from typing import Dict, List, Any

from config import TEST_QUESTIONS_PATH, EVAL_RESULTS_PATH, MODEL, USE_MOCK
from llm_client import call_llm


def load_test_questions() -> List[Dict[str, Any]]:
    """
    读取测试问题集。

    返回：
        一个列表，每个元素是一道测试题。
    """

    if not TEST_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"测试问题集不存在：{TEST_QUESTIONS_PATH}")

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_answer(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    使用关键词命中方式评估回答。

    参数：
        answer: 模型回答；
        expected_keywords: 预期关键词列表。

    返回：
        命中结果、命中数量、总关键词数量和得分。
    """

    hit_keywords = []
    missed_keywords = []

    for keyword in expected_keywords:
        if keyword in answer:
            hit_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    total = len(expected_keywords)
    hit_count = len(hit_keywords)

    if total == 0:
        score = 0.0
    else:
        score = round(hit_count / total, 2)

    return {
        "hit_keywords": hit_keywords,
        "missed_keywords": missed_keywords,
        "hit_count": hit_count,
        "total_keywords": total,
        "score": score
    }


def run_evaluation() -> Dict[str, Any]:
    """
    运行完整评测流程。

    流程：
        1. 读取测试问题；
        2. 针对每个问题调用 call_llm；
        3. 检查回答是否包含预期关键词；
        4. 统计每题得分和平均分；
        5. 保存评测结果。
    """

    test_questions = load_test_questions()

    results = []
    total_score = 0.0

    for item in test_questions:
        question_id = item.get("id")
        question = item.get("question", "")
        mode = item.get("mode", "general")
        expected_keywords = item.get("expected_keywords", [])

        answer = call_llm(question, mode=mode)
        eval_result = evaluate_answer(answer, expected_keywords)

        total_score += eval_result["score"]

        results.append({
            "id": question_id,
            "question": question,
            "mode": mode,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "hit_keywords": eval_result["hit_keywords"],
            "missed_keywords": eval_result["missed_keywords"],
            "score": eval_result["score"]
        })

    question_count = len(test_questions)

    if question_count == 0:
        average_score = 0.0
    else:
        average_score = round(total_score / question_count, 2)

    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "mock": USE_MOCK,
        "question_count": question_count,
        "average_score": average_score,
        "results": results
    }

    save_eval_results(summary)

    return summary


def save_eval_results(summary: Dict[str, Any]) -> None:
    """
    保存评测结果到 results/eval_results.json。
    """

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_eval_results() -> Dict[str, Any]:
    """
    读取最近一次评测结果。
    """

    if not EVAL_RESULTS_PATH.exists() or EVAL_RESULTS_PATH.stat().st_size == 0:
        return {
            "message": "暂时没有评测结果，请先运行评测。"
        }

    try:
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except JSONDecodeError:
        return {
            "message": "评测结果文件格式损坏，请重新运行评测。"
        }


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, ensure_ascii=False, indent=2))