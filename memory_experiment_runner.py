import csv
import json
import os
import time
from typing import Any, Dict, List

from agent_memory import ConversationMemory
from experiment_runner import (
    EVAL_DIR,
    check_no_context_reject,
    check_source_hit,
    normalize_sources,
)
from rag_agent import run_rag_agent


MEMORY_TEST_FILE = "data/memory_eval_questions.json"
MEMORY_EVAL_FILE = "eval_results/agent_memory_eval.csv"
MEMORY_SUMMARY_FILE = "eval_results/agent_memory_summary.json"


def load_memory_test_cases(file_path: str = MEMORY_TEST_FILE) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_resolved_keywords(
    resolved_question: str,
    expected_keywords: List[str],
) -> bool:
    if not expected_keywords:
        return True

    return all(keyword in resolved_question for keyword in expected_keywords)


def check_topic_follow(result: Dict[str, Any], expected_topic: str) -> bool:
    if not expected_topic:
        return True

    memory_snapshot = result.get("memory_snapshot") or {}
    current_topic = memory_snapshot.get("current_topic", "")
    current_topics = memory_snapshot.get("current_topics", [])
    resolved_question = result.get("resolved_question", "")

    return (
        expected_topic == current_topic
        or expected_topic in current_topics
        or expected_topic in resolved_question
    )


def calc_rate(values: List[Any]) -> float:
    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return sum(1 for value in valid_values if value is True) / len(valid_values)


def save_csv(rows: List[Dict[str, Any]], output_file: str):
    if not rows:
        return

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_memory_eval(
    output_file: str = MEMORY_EVAL_FILE,
    top_k: int = 3,
    candidate_k: int = 10,
    max_rewrites: int = 1,
    use_rerank: bool = True,
    max_retries: int = 2,
    retry_sleep_seconds: int = 3,
) -> Dict[str, Any]:
    os.makedirs(EVAL_DIR, exist_ok=True)

    test_cases = load_memory_test_cases()
    rows: List[Dict[str, Any]] = []

    for case in test_cases:
        case_id = case.get("case_id")
        title = case.get("title", "")
        memory = ConversationMemory()

        print("=" * 80)
        print(f"正在评测记忆 Case：{case_id} | {title}")

        for turn in case.get("turns", []):
            turn_id = turn.get("turn_id")
            question = turn.get("question", "")

            print(f"- Turn {turn_id}: {question}")

            try:
                for attempt in range(max_retries + 1):
                    try:
                        result = run_rag_agent(
                            question=question,
                            top_k=top_k,
                            candidate_k=candidate_k,
                            max_rewrites=max_rewrites,
                            use_rerank=use_rerank,
                            memory=memory,
                        )
                        break
                    except Exception as retry_error:
                        if attempt >= max_retries:
                            raise

                        print(
                            f"记忆评测调用失败，"
                            f"{retry_sleep_seconds} 秒后重试 "
                            f"({attempt + 1}/{max_retries})：{retry_error}"
                        )
                        time.sleep(retry_sleep_seconds)

                sources = result.get("sources", [])
                retrieved_sources = normalize_sources(sources)
                expected_source = turn.get("expected_source")
                expected_memory_used = turn.get("expected_memory_used")
                expect_no_context_reject = turn.get("expect_no_context_reject")
                answer = result.get("answer", "")

                source_hit = check_source_hit(expected_source, retrieved_sources)
                memory_used_hit = result.get("memory_used") == expected_memory_used
                resolved_hit = check_resolved_keywords(
                    resolved_question=result.get("resolved_question", ""),
                    expected_keywords=turn.get("expected_resolved_keywords", []),
                )
                topic_follow_hit = check_topic_follow(
                    result=result,
                    expected_topic=turn.get("expected_topic", ""),
                )

                if expect_no_context_reject is True:
                    no_context_reject = check_no_context_reject("missing", answer)
                else:
                    no_context_reject = None

                rows.append({
                    "case_id": case_id,
                    "title": title,
                    "turn_id": turn_id,
                    "question": question,
                    "resolved_question": result.get("resolved_question"),
                    "expected_topic": turn.get("expected_topic"),
                    "current_topic": (result.get("memory_snapshot") or {}).get("current_topic"),
                    "expected_memory_used": expected_memory_used,
                    "memory_used": result.get("memory_used"),
                    "memory_used_hit": memory_used_hit,
                    "expected_resolved_keywords": "|".join(turn.get("expected_resolved_keywords", [])),
                    "memory_rewrite_hit": resolved_hit,
                    "expected_source": expected_source,
                    "retrieved_sources": "|".join(retrieved_sources),
                    "source_hit": source_hit,
                    "topic_follow_hit": topic_follow_hit,
                    "expect_no_context_reject": expect_no_context_reject,
                    "no_context_reject": no_context_reject,
                    "query_type": result.get("query_type"),
                    "retriever_mode": result.get("retriever_mode"),
                    "context_sufficient": result.get("context_sufficient"),
                    "answer_length": len(answer),
                })

            except Exception as exc:
                rows.append({
                    "case_id": case_id,
                    "title": title,
                    "turn_id": turn_id,
                    "question": question,
                    "resolved_question": "",
                    "expected_topic": turn.get("expected_topic"),
                    "current_topic": "",
                    "expected_memory_used": turn.get("expected_memory_used"),
                    "memory_used": False,
                    "memory_used_hit": False,
                    "expected_resolved_keywords": "|".join(turn.get("expected_resolved_keywords", [])),
                    "memory_rewrite_hit": False,
                    "expected_source": turn.get("expected_source"),
                    "retrieved_sources": "",
                    "source_hit": False if turn.get("expected_source") is not None else None,
                    "topic_follow_hit": False,
                    "expect_no_context_reject": turn.get("expect_no_context_reject"),
                    "no_context_reject": False if turn.get("expect_no_context_reject") else None,
                    "query_type": "",
                    "retriever_mode": "",
                    "context_sufficient": False,
                    "answer_length": 0,
                })
                print(f"记忆评测失败：{exc}")

    save_csv(rows, output_file)

    summary = {
        "case_count": len(test_cases),
        "turn_count": len(rows),
        "memory_used_accuracy": calc_rate([row["memory_used_hit"] for row in rows]),
        "memory_rewrite_hit_rate": calc_rate([row["memory_rewrite_hit"] for row in rows]),
        "source_hit_rate": calc_rate([row["source_hit"] for row in rows]),
        "topic_follow_hit_rate": calc_rate([row["topic_follow_hit"] for row in rows]),
        "no_context_reject_rate": calc_rate([row["no_context_reject"] for row in rows]),
    }

    with open(MEMORY_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n记忆评测完成：", output_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("记忆评测汇总：", MEMORY_SUMMARY_FILE)

    return summary


def main():
    run_memory_eval()


if __name__ == "__main__":
    main()
