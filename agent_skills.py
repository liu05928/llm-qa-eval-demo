from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

from agent_memory import ensure_memory
from agent_state import AgentTraceStep
from context_guard import verify_answer_claims
from llm_client import call_llm
from prompt_templates import build_rag_prompt
from rag_agent import (
    build_no_context_answer,
    classify_query,
    judge_context,
    rewrite_query_with_metadata,
    select_strategy,
)
from rag_pipeline import (
    build_sources,
    format_context,
    retrieve_chunks,
)


SkillOutput = Tuple[str, Dict[str, Any], Dict[str, Any]]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_agent_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    """Populate optional fields used by the LangGraph Skills runner."""

    state.setdefault("agent_trace", [])
    state.setdefault("skill_trace", [])
    state.setdefault("graph_trace", [])
    state.setdefault("errors", [])
    state.setdefault("rewritten_queries", [])
    state.setdefault("rewrite_steps", [])
    state.setdefault("retrieved_chunks", [])
    state.setdefault("small_retrieved_chunks", [])
    state.setdefault("retrieval_logs", [])
    state.setdefault("sources", [])
    state.setdefault("long_context", {})
    state.setdefault("context_sufficient", False)
    state.setdefault("context_reason", "")
    state.setdefault("context_coverage", 0.0)
    state.setdefault("support_level", "unsupported")
    state.setdefault("evidence_score", 0.0)
    state.setdefault("guard_mode", "v2")
    state.setdefault("guard_details", {})
    state.setdefault("claim_verification", {})
    state.setdefault("memory_used", False)
    state.setdefault("memory_reason", "")
    state.setdefault("memory_snapshot", {})
    state.setdefault("rewrite_count", 0)
    state.setdefault("retriever_mode", "dense_rerank")
    state.setdefault("query_type", "general")
    state.setdefault("query_type_label", "普通问答")
    state.setdefault("query_reason", "")
    state.setdefault("current_query", state.get("question", ""))
    state.setdefault("resolved_question", state.get("question", ""))
    return state


def run_skill(
    state: Dict[str, Any],
    node: str,
    action: str,
    tool_name: str,
    input_summary: str,
    handler: Callable[[], SkillOutput],
) -> Dict[str, Any]:
    """Run a Skill and append observable trace data without exposing hidden reasoning."""

    ensure_agent_defaults(state)
    start = perf_counter()
    status = "success"
    output_summary = ""
    metadata: Dict[str, Any] = {}

    try:
        output_summary, updates, metadata = handler()
        state.update(updates)
        return state
    except Exception as exc:
        status = "error"
        output_summary = str(exc)
        state.setdefault("errors", []).append(
            {
                "node": node,
                "tool_name": tool_name,
                "message": str(exc),
                "time": _now(),
            }
        )
        raise
    finally:
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        trace_step = AgentTraceStep(
            node=node,
            action=action,
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            elapsed_ms=elapsed_ms,
            metadata=metadata,
        )
        state.setdefault("agent_trace", []).append(trace_step.to_dict())
        state.setdefault("skill_trace", []).append(
            {
                "node": node,
                "tool_name": tool_name,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "metadata": metadata,
            }
        )
        state.setdefault("graph_trace", []).append(node)


def resolve_memory_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        memory = ensure_memory(state.get("memory"))

        if not memory:
            return (
                "未启用会话记忆。",
                {
                    "memory": None,
                    "memory_used": False,
                    "memory_reason": "未传入会话记忆。",
                    "resolved_question": state["question"],
                    "current_query": state["question"],
                    "memory_snapshot": {},
                },
                {"memory_enabled": False},
            )

        resolution = memory.resolve_question(state["question"])
        resolved_question = resolution.get("resolved_question", state["question"])

        return (
            resolution.get("memory_reason", ""),
            {
                "memory": memory,
                "memory_used": bool(resolution.get("memory_used")),
                "memory_reason": resolution.get("memory_reason", ""),
                "resolved_question": resolved_question,
                "current_query": resolved_question,
                "memory_snapshot": memory.to_dict(),
            },
            {
                "memory_enabled": True,
                "memory_used": bool(resolution.get("memory_used")),
                "resolved_question": resolved_question,
            },
        )

    return run_skill(
        state=state,
        node="resolve_memory",
        action="使用 session 级短期记忆补全指代问题",
        tool_name="ConversationMemory.resolve_question",
        input_summary=state.get("question", "")[:120],
        handler=handler,
    )


def classify_query_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        query_info = classify_query(state.get("current_query", state["question"]))

        return (
            query_info.get("reason", ""),
            {
                "query_type": query_info.get("query_type", "general"),
                "query_type_label": query_info.get("query_type_label", "普通问答"),
                "query_reason": query_info.get("reason", ""),
            },
            {
                "query_type": query_info.get("query_type", "general"),
                "label": query_info.get("query_type_label", "普通问答"),
            },
        )

    return run_skill(
        state=state,
        node="classify_query",
        action="识别问题类型",
        tool_name="classify_query",
        input_summary=state.get("current_query", "")[:120],
        handler=handler,
    )


def select_strategy_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        retriever_mode = select_strategy(state.get("query_type", "general"))

        return (
            f"选择检索策略：{retriever_mode}",
            {"retriever_mode": retriever_mode},
            {
                "query_type": state.get("query_type"),
                "retriever_mode": retriever_mode,
            },
        )

    return run_skill(
        state=state,
        node="select_strategy",
        action="根据问题类型选择检索策略",
        tool_name="select_strategy",
        input_summary=state.get("query_type", "general"),
        handler=handler,
    )


def maybe_rewrite_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        should_rewrite = (
            state.get("query_type") == "vague"
            and not state.get("memory_used")
            and state.get("rewrite_count", 0) < state.get("max_rewrites", 1)
        )

        if not should_rewrite:
            return (
                "当前问题不需要预改写。",
                {},
                {
                    "rewritten": False,
                    "reason": "not_required",
                },
            )

        rewritten_query, rewrite_metadata = rewrite_query_with_metadata(
            question=state.get("current_query", state["question"]),
            query_type=state.get("query_type", "vague"),
        )
        rewritten_query = rewritten_query or state.get("current_query", state["question"])
        rewrite_info = {
            "rewritten_query": rewritten_query,
            **rewrite_metadata,
        }

        rewritten_queries = list(state.get("rewritten_queries", []))
        rewrite_steps = list(state.get("rewrite_steps", []))
        rewritten_queries.append(rewritten_query)
        rewrite_steps.append(rewrite_info)

        return (
            f"模糊问题预改写为：{rewritten_query}",
            {
                "current_query": rewritten_query,
                "rewritten_queries": rewritten_queries,
                "rewrite_steps": rewrite_steps,
                "rewrite_count": state.get("rewrite_count", 0) + 1,
            },
            {
                "rewritten": True,
                "rewritten_query": rewritten_query,
                "rewrite_strategy": rewrite_info.get("rewrite_strategy"),
            },
        )

    return run_skill(
        state=state,
        node="maybe_rewrite",
        action="必要时进行 Query Rewrite",
        tool_name="rewrite_query_with_metadata",
        input_summary=state.get("current_query", "")[:120],
        handler=handler,
    )


def retrieve_context_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        retrieved_chunks, retrieval_log = retrieve_chunks(
            question=state.get("current_query", state["question"]),
            top_k=state.get("top_k", 3),
            retriever_mode=state.get("retriever_mode", "dense_rerank"),
            candidate_k=state.get("candidate_k", 10),
            use_rerank=state.get("use_rerank", True),
            context_mode=state.get("context_mode", "small_to_big"),
        )
        retrieval_logs = list(state.get("retrieval_logs", []))
        retrieval_logs.append(retrieval_log)

        return (
            f"检索到 {len(retrieved_chunks)} 个最终上下文片段。",
            {
                "retrieved_chunks": retrieved_chunks,
                "small_retrieved_chunks": retrieval_log.get("small_final_context", []),
                "long_context": retrieval_log.get("long_context", {}),
                "retrieval_logs": retrieval_logs,
            },
            {
                "retriever_mode": state.get("retriever_mode"),
                "context_mode": state.get("context_mode"),
                "retrieved_count": len(retrieved_chunks),
                "candidate_k": state.get("candidate_k"),
            },
        )

    return run_skill(
        state=state,
        node="retrieve_context",
        action="调用 RAG 检索工具获取上下文",
        tool_name="retrieve_chunks",
        input_summary=state.get("current_query", "")[:120],
        handler=handler,
    )


def judge_context_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        context_info = judge_context(
            original_question=state.get("resolved_question", state["question"]),
            current_query=state.get("current_query", state["question"]),
            chunks=state.get("retrieved_chunks", []),
            query_type=state.get("query_type", "general"),
            guard_mode=state.get("guard_mode", "v2"),
        )

        return (
            context_info.get("reason", ""),
            {
                "context_sufficient": bool(context_info.get("context_sufficient")),
                "context_reason": context_info.get("reason", ""),
                "context_coverage": context_info.get("coverage", 0.0),
                "support_level": context_info.get("support_level", "unsupported"),
                "evidence_score": context_info.get("evidence_score", 0.0),
                "guard_details": context_info.get("guard_details", {}),
            },
            {
                "sufficient": bool(context_info.get("context_sufficient")),
                "coverage": context_info.get("coverage", 0.0),
                "support_level": context_info.get("support_level", "unsupported"),
                "evidence_score": context_info.get("evidence_score", 0.0),
                "guard_mode": state.get("guard_mode", "v2"),
            },
        )

    return run_skill(
        state=state,
        node="judge_context",
        action="判断检索上下文是否足够回答",
        tool_name="judge_context",
        input_summary=f"{len(state.get('retrieved_chunks', []))} chunks",
        handler=handler,
    )


def retry_rewrite_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        rewritten_query, rewrite_metadata = rewrite_query_with_metadata(
            question=state.get("current_query", state["question"]),
            query_type=state.get("query_type", "general"),
        )
        rewritten_query = rewritten_query or state.get("current_query", state["question"])
        rewrite_info = {
            "rewritten_query": rewritten_query,
            **rewrite_metadata,
        }

        rewritten_queries = list(state.get("rewritten_queries", []))
        rewrite_steps = list(state.get("rewrite_steps", []))
        rewritten_queries.append(rewritten_query)
        rewrite_steps.append(rewrite_info)

        return (
            f"上下文不足，改写后重检索：{rewritten_query}",
            {
                "current_query": rewritten_query,
                "rewritten_queries": rewritten_queries,
                "rewrite_steps": rewrite_steps,
                "rewrite_count": state.get("rewrite_count", 0) + 1,
            },
            {
                "rewritten_query": rewritten_query,
                "rewrite_strategy": rewrite_info.get("rewrite_strategy"),
                "rewrite_count": state.get("rewrite_count", 0) + 1,
            },
        )

    return run_skill(
        state=state,
        node="rewrite_query",
        action="上下文不足时进行二次 Query Rewrite",
        tool_name="rewrite_query_with_metadata",
        input_summary=state.get("current_query", "")[:120],
        handler=handler,
    )


def generate_answer_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        if not state.get("context_sufficient"):
            answer = build_no_context_answer(
                question=state.get("resolved_question", state["question"]),
                context_reason=state.get("context_reason", "上下文不足。"),
            )

            return (
                "上下文不足，生成拒答回答。",
                {
                    "answer": answer,
                    "sources": [],
                    "claim_verification": {
                        "enabled": True,
                        "claim_count": 0,
                        "supported_claim_count": 0,
                        "unsupported_claim_count": 0,
                        "claims": [],
                        "status": "skipped_no_context",
                    },
                },
                {
                    "no_context": True,
                    "source_count": 0,
                },
            )

        context = format_context(state.get("retrieved_chunks", []))
        rag_prompt = build_rag_prompt(
            question=state.get("resolved_question", state["question"]),
            context=context,
            guard_details=state.get("guard_details", {}),
        )
        answer = call_llm(
            question=rag_prompt,
            mode="education",
        )
        sources = build_sources(state.get("retrieved_chunks", []))
        claim_verification = verify_answer_claims(
            answer=answer,
            chunks=state.get("retrieved_chunks", []),
        )

        return (
            f"生成回答，引用 {len(sources)} 个来源。",
            {
                "answer": answer,
                "sources": sources,
                "claim_verification": claim_verification,
            },
            {
                "no_context": False,
                "source_count": len(sources),
                "answer_chars": len(answer or ""),
                "claim_verification_status": claim_verification.get("status"),
            },
        )

    return run_skill(
        state=state,
        node="generate_answer",
        action="基于检索上下文生成最终回答",
        tool_name="build_rag_prompt+call_llm",
        input_summary=state.get("resolved_question", "")[:120],
        handler=handler,
    )


def update_memory_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        memory = ensure_memory(state.get("memory"))

        if not memory:
            return (
                "未启用会话记忆，跳过更新。",
                {"memory_snapshot": {}},
                {"memory_enabled": False},
            )

        memory.update_from_result(
            question=state.get("question", ""),
            resolved_question=state.get("resolved_question", state.get("question", "")),
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            query_type=state.get("query_type", "general"),
            retriever_mode=state.get("retriever_mode", ""),
            rewritten_queries=state.get("rewritten_queries", []),
        )

        return (
            "会话记忆已更新。",
            {
                "memory": memory,
                "memory_snapshot": memory.to_dict(),
            },
            {
                "current_topic": memory.current_topic,
                "recent_turn_count": len(memory.recent_turns),
            },
        )

    return run_skill(
        state=state,
        node="update_memory",
        action="写入 session 级短期记忆",
        tool_name="ConversationMemory.update_from_result",
        input_summary=state.get("resolved_question", "")[:120],
        handler=handler,
    )


def finalize_response_skill(state: Dict[str, Any]) -> Dict[str, Any]:
    def handler() -> SkillOutput:
        return (
            "Agent 执行完成，已整理可观测结果。",
            {
                "finalized_at": _now(),
            },
            {
                "trace_steps": len(state.get("agent_trace", [])),
                "sources": len(state.get("sources", [])),
            },
        )

    return run_skill(
        state=state,
        node="finalize_response",
        action="整理 Agent 输出",
        tool_name="finalize_response",
        input_summary=state.get("question", "")[:120],
        handler=handler,
    )
