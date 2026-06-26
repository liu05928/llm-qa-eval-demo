import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


os.environ.setdefault("USE_MOCK", "true")

from fastapi.testclient import TestClient

from agent_memory import ConversationMemory
from app import app
from config import USE_MOCK
from experiment_runner import (
    check_has_citation,
    check_keyword_hit,
    check_no_context_reject,
    check_source_hit,
    collect_keyword_hits,
    load_test_questions,
    normalize_sources,
)
from langgraph_agent import LANGGRAPH_AVAILABLE, run_langgraph_rag_agent
from rag_agent import run_rag_agent
from runtime_controls import (
    check_rate_limit,
    get_runtime_controls_status,
)


EVAL_DIR = Path("eval_results")
SUMMARY_FILE = EVAL_DIR / "engineering_experiment_summary.json"
REPORT_FILE = EVAL_DIR / "engineering_experiment_report.md"
DETAIL_FILE = EVAL_DIR / "engineering_experiment_details.csv"


CORE_GRAPH_NODES = [
    "resolve_memory",
    "classify_query",
    "select_strategy",
    "retrieve_context",
    "judge_context",
    "generate_answer",
    "update_memory",
    "finalize_response",
]


def calc_rate(values: List[Optional[bool]]) -> float:
    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return sum(1 for value in valid_values if value is True) / len(valid_values)


def save_csv(rows: List[Dict[str, Any]], output_file: Path):
    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_detail(rows: List[Dict[str, Any]], section: str, data: Dict[str, Any]):
    row = {"section": section}
    row.update(data)
    rows.append(row)


def run_agent_engine_experiment(detail_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    questions = [
        "什么是 RAG？",
        "RAG 和普通大模型问答有什么区别？",
        "这个技术有什么用？",
        "资料里有没有介绍 Rerank 最新商业定价？",
    ]
    rows: List[Dict[str, Any]] = []

    for question in questions:
        for engine in ["local", "langgraph"]:
            memory = ConversationMemory()

            if engine == "local":
                result = run_rag_agent(
                    question=question,
                    top_k=2,
                    candidate_k=5,
                    max_rewrites=1,
                    use_rerank=False,
                    memory=memory,
                )
                result["agent_engine"] = "local"
            else:
                result = run_langgraph_rag_agent(
                    question=question,
                    top_k=2,
                    candidate_k=5,
                    max_rewrites=1,
                    use_rerank=False,
                    memory=memory,
                )

            graph_trace = result.get("graph_trace") or []
            graph_trace_complete = (
                engine == "local"
                or all(node in graph_trace for node in CORE_GRAPH_NODES)
            )
            trace_count = len(result.get("agent_trace") or [])
            skill_trace_count = len(result.get("skill_trace") or [])
            answer_nonempty = bool(result.get("answer"))
            passed = answer_nonempty and trace_count >= 5 and graph_trace_complete

            row = {
                "question": question,
                "engine": result.get("agent_engine", engine),
                "query_type": result.get("query_type"),
                "retriever_mode": result.get("retriever_mode"),
                "context_sufficient": result.get("context_sufficient"),
                "answer_nonempty": answer_nonempty,
                "source_count": len(result.get("sources") or []),
                "agent_trace_count": trace_count,
                "skill_trace_count": skill_trace_count,
                "graph_trace_complete": graph_trace_complete,
                "graph_trace": " -> ".join(graph_trace),
                "passed": passed,
            }
            rows.append(row)
            append_detail(detail_rows, "agent_engine", row)

    langgraph_rows = [row for row in rows if row["engine"] == "langgraph"]

    return {
        "question_count": len(questions),
        "row_count": len(rows),
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "pass_rate": calc_rate([row["passed"] for row in rows]),
        "langgraph_graph_trace_pass_rate": calc_rate(
            [row["graph_trace_complete"] for row in langgraph_rows]
        ),
        "langgraph_avg_skill_trace_count": (
            sum(row["skill_trace_count"] for row in langgraph_rows) / len(langgraph_rows)
            if langgraph_rows
            else 0
        ),
    }


def run_memory_api_experiment(detail_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = TestClient(app)
    session_id = f"engineering-memory-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    turns = [
        {
            "turn_id": 1,
            "question": "什么是 RAG？",
            "expected_memory_used": False,
            "expected_resolved_keywords": ["RAG"],
        },
        {
            "turn_id": 2,
            "question": "它为什么可以减少幻觉？",
            "expected_memory_used": True,
            "expected_resolved_keywords": ["RAG", "幻觉"],
        },
    ]
    rows: List[Dict[str, Any]] = []

    for turn in turns:
        response = client.post(
            "/agent/chat",
            json={
                "question": turn["question"],
                "session_id": session_id,
                "agent_engine": "langgraph",
                "top_k": 2,
                "candidate_k": 5,
                "max_rewrites": 1,
                "use_rerank": False,
                "context_mode": "small_to_big",
                "enable_rate_limit": False,
            },
        )
        data = response.json()
        resolved_question = data.get("resolved_question", "")
        memory_used = data.get("memory_used")
        resolved_hit = all(
            keyword in resolved_question
            for keyword in turn["expected_resolved_keywords"]
        )
        memory_used_hit = memory_used == turn["expected_memory_used"]

        row = {
            "turn_id": turn["turn_id"],
            "status_code": response.status_code,
            "question": turn["question"],
            "resolved_question": resolved_question,
            "expected_memory_used": turn["expected_memory_used"],
            "memory_used": memory_used,
            "memory_used_hit": memory_used_hit,
            "resolved_hit": resolved_hit,
            "agent_engine": data.get("agent_engine"),
            "session_id": data.get("session_id"),
            "graph_trace_count": len(data.get("graph_trace") or []),
            "passed": response.status_code == 200 and memory_used_hit and resolved_hit,
        }
        rows.append(row)
        append_detail(detail_rows, "memory_api", row)

    session_response = client.get(f"/agent/session/{session_id}")
    session_data = session_response.json() if session_response.status_code == 200 else {}
    memory_snapshot = session_data.get("memory", {})
    session_snapshot_ok = (
        session_response.status_code == 200
        and len(memory_snapshot.get("recent_turns", [])) >= 2
    )

    client.delete(f"/agent/session/{session_id}")

    return {
        "session_id": session_id,
        "turn_count": len(rows),
        "pass_rate": calc_rate([row["passed"] for row in rows]),
        "memory_used_accuracy": calc_rate([row["memory_used_hit"] for row in rows]),
        "resolved_hit_rate": calc_rate([row["resolved_hit"] for row in rows]),
        "session_snapshot_ok": session_snapshot_ok,
        "session_recent_turn_count": len(memory_snapshot.get("recent_turns", [])),
    }


def run_cache_rate_limit_experiment(detail_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = TestClient(app)
    runtime_status = get_runtime_controls_status()
    redis_available = bool(runtime_status.get("redis_available"))
    result: Dict[str, Any] = {
        "redis_available": redis_available,
        "runtime_status": runtime_status,
    }

    if not redis_available:
        response = client.post(
            "/agent/chat",
            json={
                "question": "什么是 RAG？",
                "agent_engine": "langgraph",
                "top_k": 1,
                "candidate_k": 3,
                "max_rewrites": 1,
                "use_rerank": False,
                "enable_cache": True,
                "enable_rate_limit": True,
            },
        )
        data = response.json()
        fail_open_passed = (
            response.status_code == 200
            and data.get("cache_status") == "unavailable"
            and (data.get("rate_limit") or {}).get("enabled") is False
        )
        row = {
            "redis_available": False,
            "status_code": response.status_code,
            "cache_status": data.get("cache_status"),
            "cache_hit": data.get("cache_hit"),
            "rate_limit_enabled": (data.get("rate_limit") or {}).get("enabled"),
            "passed": fail_open_passed,
        }
        append_detail(detail_rows, "cache_rate_limit", row)
        result.update(
            {
                "mode": "redis_unavailable_fail_open",
                "cache_test": "skipped",
                "rate_limit_test": "skipped",
                "fail_open_passed": fail_open_passed,
            }
        )
        return result

    first = client.post(
        "/agent/chat",
        json={
            "question": "什么是 RAG？",
            "agent_engine": "langgraph",
            "top_k": 1,
            "candidate_k": 3,
            "max_rewrites": 1,
            "use_rerank": False,
            "enable_cache": True,
            "enable_rate_limit": False,
        },
    ).json()
    second = client.post(
        "/agent/chat",
        json={
            "question": "什么是 RAG？",
            "agent_engine": "langgraph",
            "top_k": 1,
            "candidate_k": 3,
            "max_rewrites": 1,
            "use_rerank": False,
            "enable_cache": True,
            "enable_rate_limit": False,
        },
    ).json()
    cache_passed = (
        first.get("cache_hit") is False
        and second.get("cache_hit") is True
        and second.get("cache_status") == "hit"
    )

    identifier = f"engineering-rate-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    first_allowed, first_info = check_rate_limit(identifier, limit=1)
    second_allowed, second_info = check_rate_limit(identifier, limit=1)
    rate_limit_passed = first_allowed is True and second_allowed is False

    row = {
        "redis_available": True,
        "first_cache_status": first.get("cache_status"),
        "second_cache_status": second.get("cache_status"),
        "second_cache_hit": second.get("cache_hit"),
        "cache_passed": cache_passed,
        "first_rate_allowed": first_info.get("allowed"),
        "second_rate_allowed": second_info.get("allowed"),
        "rate_limit_passed": rate_limit_passed,
        "passed": cache_passed and rate_limit_passed,
    }
    append_detail(detail_rows, "cache_rate_limit", row)

    result.update(
        {
            "mode": "redis_available",
            "cache_test": "passed" if cache_passed else "failed",
            "rate_limit_test": "passed" if rate_limit_passed else "failed",
            "cache_passed": cache_passed,
            "rate_limit_passed": rate_limit_passed,
        }
    )
    return result


def select_regression_questions() -> List[Dict[str, Any]]:
    questions = load_test_questions()
    selected: List[Dict[str, Any]] = []
    preferred_ids = {1, 5, 11, 21, 31}

    for item in questions:
        if item.get("id") in preferred_ids:
            selected.append(item)

    missing_item = next(
        (item for item in questions if item.get("question_type") == "missing"),
        None,
    )

    if missing_item:
        selected.append(missing_item)

    return selected


def run_rag_api_regression(detail_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = TestClient(app)
    rows: List[Dict[str, Any]] = []

    for item in select_regression_questions():
        response = client.post(
            "/rag_chat",
            json={
                "question": item["question"],
                "top_k": 3,
                "candidate_k": 10,
                "retriever_mode": "bm25_hybrid",
                "context_mode": "small_to_big",
                "use_rerank": True,
            },
        )
        data = response.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        retrieved_sources = normalize_sources(sources)
        source_hit = check_source_hit(item.get("expected_source"), retrieved_sources)
        expected_keywords = item.get("expected_keywords", [])
        keyword_hits = collect_keyword_hits(expected_keywords, answer)
        keyword_hit = check_keyword_hit(expected_keywords, answer)
        has_citation = check_has_citation(sources)
        no_context_reject = check_no_context_reject(item.get("question_type"), answer)
        if item.get("question_type") == "missing" and no_context_reject:
            keyword_hit = True
            keyword_hits.append("no_context_reject")
        route_ok = response.status_code == 200 and bool(answer)

        row = {
            "id": item.get("id"),
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "status_code": response.status_code,
            "route_ok": route_ok,
            "expected_source": item.get("expected_source"),
            "retrieved_sources": "|".join(retrieved_sources),
            "source_hit": source_hit,
            "expected_keywords": "|".join(expected_keywords),
            "keyword_hits": "|".join(keyword_hits),
            "keyword_hit": keyword_hit,
            "has_citation": has_citation,
            "no_context_reject": no_context_reject,
            "retrieved_chunk_count": len(data.get("retrieved_chunks") or []),
            "answer_length": len(answer),
        }
        rows.append(row)
        append_detail(detail_rows, "rag_api_regression", row)

    source_miss_cases = [
        {
            "id": row["id"],
            "question": row["question"],
            "expected_source": row["expected_source"],
            "retrieved_sources": row["retrieved_sources"],
        }
        for row in rows
        if row["source_hit"] is False
    ]

    return {
        "question_count": len(rows),
        "route_ok_rate": calc_rate([row["route_ok"] for row in rows]),
        "source_hit_rate": calc_rate([row["source_hit"] for row in rows]),
        "keyword_hit_rate": calc_rate([row["keyword_hit"] for row in rows]),
        "has_citation_rate": calc_rate([row["has_citation"] for row in rows]),
        "no_context_reject_rate": calc_rate([row["no_context_reject"] for row in rows]),
        "source_miss_count": len(source_miss_cases),
        "source_miss_cases": source_miss_cases,
        "avg_answer_length": (
            sum(row["answer_length"] for row in rows) / len(rows)
            if rows
            else 0
        ),
    }


def write_report(summary: Dict[str, Any]):
    lines = [
        "# Engineering Experiment Report",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- USE_MOCK: `{summary['use_mock']}`",
        f"- LangGraph available: `{summary['langgraph_available']}`",
        "",
        "## 1. Agent Engine",
        "",
        f"- Questions: {summary['agent_engine_compare']['question_count']}",
        f"- Pass rate: {summary['agent_engine_compare']['pass_rate']:.2%}",
        f"- LangGraph graph trace pass rate: {summary['agent_engine_compare']['langgraph_graph_trace_pass_rate']:.2%}",
        f"- LangGraph avg skill trace count: {summary['agent_engine_compare']['langgraph_avg_skill_trace_count']:.2f}",
        "",
        "## 2. Session Memory API",
        "",
        f"- Pass rate: {summary['memory_api']['pass_rate']:.2%}",
        f"- Memory used accuracy: {summary['memory_api']['memory_used_accuracy']:.2%}",
        f"- Resolved question hit rate: {summary['memory_api']['resolved_hit_rate']:.2%}",
        f"- Session snapshot OK: `{summary['memory_api']['session_snapshot_ok']}`",
        "",
        "## 3. Cache And Rate Limit",
        "",
        f"- Redis available: `{summary['cache_rate_limit']['redis_available']}`",
        f"- Mode: `{summary['cache_rate_limit']['mode']}`",
        f"- Cache test: `{summary['cache_rate_limit']['cache_test']}`",
        f"- Rate limit test: `{summary['cache_rate_limit']['rate_limit_test']}`",
        "",
        "## 4. RAG API Regression",
        "",
        f"- Questions: {summary['rag_api_regression']['question_count']}",
        f"- Route OK rate: {summary['rag_api_regression']['route_ok_rate']:.2%}",
        f"- Source hit rate: {summary['rag_api_regression']['source_hit_rate']:.2%}",
        f"- Keyword hit rate: {summary['rag_api_regression']['keyword_hit_rate']:.2%}",
        f"- Citation rate: {summary['rag_api_regression']['has_citation_rate']:.2%}",
        f"- No-context reject rate: {summary['rag_api_regression']['no_context_reject_rate']:.2%}",
        f"- Strict source miss count: {summary['rag_api_regression']['source_miss_count']}",
        "",
        "## Conclusion",
        "",
        "本轮实验重点验证 LangGraph Agent 编排、session 级记忆、缓存限流运行状态和原 RAG API 回归稳定性。Redis 不可用时系统会 fail-open，不影响 Agent/RAG 主链路运行。",
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def run_engineering_experiments() -> Dict[str, Any]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows: List[Dict[str, Any]] = []

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "use_mock": USE_MOCK,
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "agent_engine_compare": run_agent_engine_experiment(detail_rows),
        "memory_api": run_memory_api_experiment(detail_rows),
        "cache_rate_limit": run_cache_rate_limit_experiment(detail_rows),
        "rag_api_regression": run_rag_api_regression(detail_rows),
    }

    save_csv(detail_rows, DETAIL_FILE)
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary)

    return summary


def main():
    summary = run_engineering_experiments()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {SUMMARY_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Details: {DETAIL_FILE}")


if __name__ == "__main__":
    main()
