import json
from pathlib import Path

from rag_pipeline import rag_answer


TEST_FILE = Path("data/rag_test_questions.json")
RESULTS_DIR = Path("results")
RAG_EVAL_RESULT_FILE = RESULTS_DIR / "rag_eval_results.json"


def load_rag_test_questions(test_file: Path = TEST_FILE):
    """
    读取 RAG 测试问题集。
    """

    if not test_file.exists():
        raise FileNotFoundError(f"RAG 测试集不存在: {test_file}")

    with test_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_source_hit(sources, expected_source: str) -> bool:
    """
    检查检索来源是否命中预期文档。
    """

    source_names = [item["source"] for item in sources]

    return expected_source in source_names


def check_keyword_hit(answer: str, expected_keywords: list[str]) -> bool:
    """
    检查回答中是否包含任意预期关键词。

    这里先采用简单策略：
    只要命中一个关键词，就算 keyword_hit = True。
    """

    return any(keyword in answer for keyword in expected_keywords)


def evaluate_rag(top_k: int = 3):
    """
    运行 RAG 简单评测。

    评测指标：
    1. source_hit：检索结果是否包含预期来源文档；
    2. keyword_hit：回答是否包含预期关键词。
    """

    test_questions = load_rag_test_questions()

    results = []

    source_hit_count = 0
    keyword_hit_count = 0

    for item in test_questions:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_keywords"]

        print("=" * 80)
        print(f"评测问题 {item['id']}：{question}")

        rag_result = rag_answer(
            question=question,
            top_k=top_k,
        )

        sources = rag_result["sources"]
        answer = rag_result["answer"]

        source_hit = check_source_hit(
            sources=sources,
            expected_source=expected_source,
        )

        keyword_hit = check_keyword_hit(
            answer=answer,
            expected_keywords=expected_keywords,
        )

        if source_hit:
            source_hit_count += 1

        if keyword_hit:
            keyword_hit_count += 1

        result_item = {
            "id": item["id"],
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": sources,
            "expected_keywords": expected_keywords,
            "answer": answer,
            "source_hit": source_hit,
            "keyword_hit": keyword_hit,
        }

        results.append(result_item)

        print(f"预期来源：{expected_source}")
        print(f"来源命中：{source_hit}")
        print(f"关键词命中：{keyword_hit}")

    total = len(test_questions)

    summary = {
        "total": total,
        "source_hit_count": source_hit_count,
        "source_hit_rate": round(source_hit_count / total, 4) if total else 0,
        "keyword_hit_count": keyword_hit_count,
        "keyword_hit_rate": round(keyword_hit_count / total, 4) if total else 0,
    }

    eval_result = {
        "summary": summary,
        "details": results,
    }

    save_eval_results(eval_result)

    print("\n" + "=" * 80)
    print("RAG 评测完成")
    print(f"测试问题数：{total}")
    print(f"来源命中数：{source_hit_count}")
    print(f"来源命中率：{summary['source_hit_rate'] * 100:.2f}%")
    print(f"关键词命中数：{keyword_hit_count}")
    print(f"关键词命中率：{summary['keyword_hit_rate'] * 100:.2f}%")
    print(f"评测结果已保存到：{RAG_EVAL_RESULT_FILE}")

    return eval_result


def save_eval_results(eval_result: dict):
    """
    保存 RAG 评测结果。
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with RAG_EVAL_RESULT_FILE.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    evaluate_rag(top_k=3)