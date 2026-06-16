import json
import os
import time
from typing import Any, Dict, List

from experiment_runner import (
    EVAL_DIR,
    calc_summary,
    check_has_citation,
    check_keyword_hit,
    check_no_context_reject,
    check_source_hit,
    load_test_questions,
    normalize_sources,
    print_summary,
    save_csv,
)
from rag_agent import run_rag_agent


AGENT_EVAL_FILE = "eval_results/agent_rag_eval.csv"
AGENT_SUMMARY_FILE = "eval_results/agent_rag_summary.json"


def run_agent_eval(
    output_file: str = AGENT_EVAL_FILE,
    top_k: int = 3,
    candidate_k: int = 10,
    max_rewrites: int = 1,
    use_rerank: bool = True,
    max_retries: int = 2,
    retry_sleep_seconds: int = 3,
) -> Dict[str, Any]:
    """Run the single-turn evaluation set against the Single-Agent RAG workflow."""

    os.makedirs(EVAL_DIR, exist_ok=True)

    questions = load_test_questions()
    rows: List[Dict[str, Any]] = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])
        question_type = item.get("question_type", "")

        print("=" * 80)
        print(f"正在评测 Agent | 第 {qid} 题：{question}")

        try:
            for attempt in range(max_retries + 1):
                try:
                    result = run_rag_agent(
                        question=question,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        max_rewrites=max_rewrites,
                        use_rerank=use_rerank,
                    )
                    break
                except Exception as retry_error:
                    if attempt >= max_retries:
                        raise

                    print(
                        f"第 {qid} 题 Agent 调用失败，"
                        f"{retry_sleep_seconds} 秒后重试 "
                        f"({attempt + 1}/{max_retries})：{retry_error}"
                    )
                    time.sleep(retry_sleep_seconds)

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
                "query_type": result.get("query_type"),
                "query_type_label": result.get("query_type_label"),
                "retriever_mode": result.get("retriever_mode"),
                "top_k": top_k,
                "candidate_k": candidate_k,
                "max_rewrites": max_rewrites,
                "use_rerank": use_rerank,
                "context_sufficient": result.get("context_sufficient"),
                "context_coverage": result.get("context_coverage"),
                "rewritten_queries": "|".join(result.get("rewritten_queries", [])),
                "expected_source": expected_source,
                "retrieved_sources": "|".join(retrieved_sources),
                "source_hit": source_hit,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hit": keyword_hit,
                "has_citation": has_citation,
                "no_context_reject": no_context_reject,
                "retrieved_chunk_count": len(retrieved_chunks),
                "agent_trace_count": len(result.get("agent_trace", [])),
                "answer_length": len(answer),
                "answer": answer,
            })

        except Exception as exc:
            print(f"Agent 评测失败：{exc}")

            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "query_type": "",
                "query_type_label": "",
                "retriever_mode": "",
                "top_k": top_k,
                "candidate_k": candidate_k,
                "max_rewrites": max_rewrites,
                "use_rerank": use_rerank,
                "context_sufficient": False,
                "context_coverage": 0,
                "rewritten_queries": "",
                "expected_source": expected_source,
                "retrieved_sources": "",
                "source_hit": False if expected_source is not None else None,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hit": False,
                "has_citation": False,
                "no_context_reject": False if question_type == "missing" else None,
                "retrieved_chunk_count": 0,
                "agent_trace_count": 0,
                "answer_length": 0,
                "answer": f"ERROR: {exc}",
            })

    save_csv(rows, output_file)
    summary = calc_summary(rows)

    with open(AGENT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nAgent 评测完成：", output_file)
    print_summary(summary)
    print("Agent 汇总结果：", AGENT_SUMMARY_FILE)

    return summary


def main():
    run_agent_eval()


if __name__ == "__main__":
    main()
