import json
import csv
import os
from typing import List, Dict, Any

from rag_pipeline import rag_answer


TEST_FILE = "data/rag_test_questions.json"
EVAL_DIR = "eval_results"

BASELINE_FILE = "eval_results/baseline_eval.csv"
HYBRID_FILE = "eval_results/hybrid_rerank_eval.csv"
SUMMARY_FILE = "eval_results/experiment_summary.json"


def load_test_questions(file_path: str = TEST_FILE) -> List[Dict[str, Any]]:
    """读取测试集"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sources(sources):
    """把 sources 统一成 source 文件名列表"""
    if not sources:
        return []

    result = []

    for item in sources:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            source = item.get("source")
            if source:
                result.append(source)

    return result


def check_source_hit(expected_source, retrieved_sources):
    """
    判断来源是否命中。
    资料缺失类问题 expected_source 为 None，不参与来源命中率统计。
    """
    if expected_source is None:
        return None

    return expected_source in retrieved_sources


def check_keyword_hit(expected_keywords, answer):
    """
    判断关键词是否命中。
    命中一半及以上关键词，记为 True。
    """
    if not expected_keywords:
        return None

    hit_count = 0

    for keyword in expected_keywords:
        if keyword in answer:
            hit_count += 1

    return hit_count / len(expected_keywords) >= 0.5


def check_has_citation(sources):
    """判断是否返回引用来源"""
    return len(sources) > 0


def check_no_context_reject(question_type, answer):
    """
    判断资料缺失类问题是否拒答。
    只对 question_type == missing 的问题统计。
    """
    if question_type != "missing":
        return None

    refusal_phrases = [
        "无法确定",
        "资料中未提及",
        "当前资料",
        "没有提到",
        "无法从资料",
        "未提供相关信息",
        "不知道",
        "无法回答"
    ]

    return any(phrase in answer for phrase in refusal_phrases)


def run_single_eval(
    output_file: str,
    retriever_mode: str,
    top_k: int = 3,
    candidate_k: int = 10,
    use_rerank: bool = True
):
    """
    运行单组实验。
    """
    os.makedirs(EVAL_DIR, exist_ok=True)

    questions = load_test_questions()
    rows = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])
        question_type = item.get("question_type", "")

        print("=" * 80)
        print(f"正在评测：{retriever_mode} | 第 {qid} 题：{question}")

        try:
            result = rag_answer(
                question=question,
                top_k=top_k,
                retriever_mode=retriever_mode,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            retrieved_chunks = result.get("retrieved_chunks", [])

            retrieved_sources = normalize_sources(sources)

            source_hit = check_source_hit(expected_source, retrieved_sources)
            keyword_hit = check_keyword_hit(expected_keywords, answer)
            has_citation = check_has_citation(sources)
            no_context_reject = check_no_context_reject(question_type, answer)

            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "retriever_mode": retriever_mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "use_rerank": use_rerank,
                "expected_source": expected_source,
                "retrieved_sources": "|".join(retrieved_sources),
                "source_hit": source_hit,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hit": keyword_hit,
                "has_citation": has_citation,
                "no_context_reject": no_context_reject,
                "retrieved_chunk_count": len(retrieved_chunks),
                "answer_length": len(answer),
                "answer": answer
            })

        except Exception as e:
            print(f"评测失败：{e}")

            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "retriever_mode": retriever_mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "use_rerank": use_rerank,
                "expected_source": expected_source,
                "retrieved_sources": "",
                "source_hit": False if expected_source is not None else None,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hit": False,
                "has_citation": False,
                "no_context_reject": False if question_type == "missing" else None,
                "retrieved_chunk_count": 0,
                "answer_length": 0,
                "answer": f"ERROR: {e}"
            })

    save_csv(rows, output_file)
    summary = calc_summary(rows)

    print("\n评测完成：", output_file)
    print_summary(summary)

    return summary


def save_csv(rows, output_file):
    """保存 CSV"""
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calc_rate(values):
    """计算 True 比例，忽略 None"""
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        return 0

    true_count = sum(1 for v in valid_values if v is True)
    return true_count / len(valid_values)


def calc_summary(rows):
    """计算汇总指标"""
    total = len(rows)

    return {
        "total": total,
        "source_hit_rate": calc_rate([r["source_hit"] for r in rows]),
        "keyword_hit_rate": calc_rate([r["keyword_hit"] for r in rows]),
        "has_citation_rate": calc_rate([r["has_citation"] for r in rows]),
        "no_context_reject_rate": calc_rate([r["no_context_reject"] for r in rows]),
        "avg_answer_length": sum(r["answer_length"] for r in rows) / total if total else 0,
        "avg_retrieved_chunk_count": sum(r["retrieved_chunk_count"] for r in rows) / total if total else 0
    }


def print_summary(summary):
    print(f"测试问题数：{summary['total']}")
    print(f"来源命中率：{summary['source_hit_rate']:.2%}")
    print(f"关键词命中率：{summary['keyword_hit_rate']:.2%}")
    print(f"引用完整率：{summary['has_citation_rate']:.2%}")
    print(f"无资料拒答率：{summary['no_context_reject_rate']:.2%}")
    print(f"平均回答长度：{summary['avg_answer_length']:.1f}")
    print(f"平均检索片段数：{summary['avg_retrieved_chunk_count']:.1f}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("\n开始运行 Baseline：基础向量检索")
    baseline_summary = run_single_eval(
        output_file=BASELINE_FILE,
        retriever_mode="vector",
        top_k=3,
        candidate_k=3,
        use_rerank=False
    )

    print("\n开始运行 Optimized：Hybrid Search + Rerank")
    hybrid_summary = run_single_eval(
        output_file=HYBRID_FILE,
        retriever_mode="hybrid",
        top_k=3,
        candidate_k=10,
        use_rerank=True
    )

    experiment_summary = {
        "baseline": baseline_summary,
        "hybrid_rerank": hybrid_summary
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("对比实验完成")
    print(f"Baseline 结果：{BASELINE_FILE}")
    print(f"Hybrid + Rerank 结果：{HYBRID_FILE}")
    print(f"汇总结果：{SUMMARY_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()