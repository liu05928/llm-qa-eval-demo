import uuid
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, TypedDict

from agent_memory import ensure_memory
from agent_skills import (
    classify_query_skill,
    finalize_response_skill,
    generate_answer_skill,
    judge_context_skill,
    maybe_rewrite_skill,
    resolve_memory_skill,
    retry_rewrite_skill,
    retrieve_context_skill,
    select_strategy_skill,
    update_memory_skill,
)
from rag_agent import append_agent_log, run_rag_agent


try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on optional package
    END = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


class LangGraphRagState(TypedDict, total=False):
    request_id: str
    agent_engine: str
    question: str
    resolved_question: str
    current_query: str
    top_k: int
    candidate_k: int
    max_rewrites: int
    use_rerank: bool
    context_mode: str
    guard_mode: str
    query_type: str
    query_type_label: str
    query_reason: str
    retriever_mode: str
    context_sufficient: bool
    context_reason: str
    context_coverage: float
    support_level: str
    evidence_score: float
    guard_details: Dict[str, Any]
    claim_verification: Dict[str, Any]
    rewrite_count: int
    rewritten_queries: List[str]
    rewrite_steps: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    small_retrieved_chunks: List[Dict[str, Any]]
    long_context: Dict[str, Any]
    retrieval_logs: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    answer: str
    memory: Any
    memory_used: bool
    memory_reason: str
    memory_snapshot: Dict[str, Any]
    agent_trace: List[Dict[str, Any]]
    skill_trace: List[Dict[str, Any]]
    graph_trace: List[str]
    errors: List[Dict[str, Any]]
    created_at: str
    finalized_at: str


_COMPILED_GRAPH = None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _route_after_judge(state: LangGraphRagState) -> str:
    if state.get("context_sufficient"):
        return "generate_answer"

    if state.get("query_type") == "missing":
        return "generate_answer"

    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", 1)

    if rewrite_count < max_rewrites:
        return "rewrite_query"

    return "generate_answer"


def _build_graph():
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            f"LangGraph 不可用，无法构建图编排 Agent：{LANGGRAPH_IMPORT_ERROR}"
        )

    graph = StateGraph(LangGraphRagState)

    graph.add_node("resolve_memory", resolve_memory_skill)
    graph.add_node("classify_query", classify_query_skill)
    graph.add_node("select_strategy", select_strategy_skill)
    graph.add_node("maybe_rewrite", maybe_rewrite_skill)
    graph.add_node("retrieve_context", retrieve_context_skill)
    graph.add_node("judge_context", judge_context_skill)
    graph.add_node("rewrite_query", retry_rewrite_skill)
    graph.add_node("generate_answer", generate_answer_skill)
    graph.add_node("update_memory", update_memory_skill)
    graph.add_node("finalize_response", finalize_response_skill)

    graph.set_entry_point("resolve_memory")
    graph.add_edge("resolve_memory", "classify_query")
    graph.add_edge("classify_query", "select_strategy")
    graph.add_edge("select_strategy", "maybe_rewrite")
    graph.add_edge("maybe_rewrite", "retrieve_context")
    graph.add_edge("retrieve_context", "judge_context")
    graph.add_conditional_edges(
        "judge_context",
        _route_after_judge,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("generate_answer", "update_memory")
    graph.add_edge("update_memory", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()


def get_compiled_graph():
    global _COMPILED_GRAPH

    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()

    return _COMPILED_GRAPH


def _initial_state(
    question: str,
    top_k: int,
    candidate_k: int,
    max_rewrites: int,
    use_rerank: bool,
    context_mode: str,
    guard_mode: str,
    memory: Any,
) -> LangGraphRagState:
    normalized_question = question.strip()

    return {
        "request_id": f"agent-{uuid.uuid4().hex[:12]}",
        "agent_engine": "langgraph",
        "question": normalized_question,
        "resolved_question": normalized_question,
        "current_query": normalized_question,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "max_rewrites": max_rewrites,
        "use_rerank": use_rerank,
        "context_mode": context_mode,
        "guard_mode": guard_mode,
        "query_type": "general",
        "query_type_label": "普通问答",
        "query_reason": "",
        "retriever_mode": "dense_rerank",
        "context_sufficient": False,
        "context_reason": "",
        "context_coverage": 0.0,
        "support_level": "unsupported",
        "evidence_score": 0.0,
        "guard_details": {},
        "claim_verification": {},
        "rewrite_count": 0,
        "rewritten_queries": [],
        "rewrite_steps": [],
        "retrieved_chunks": [],
        "small_retrieved_chunks": [],
        "long_context": {},
        "retrieval_logs": [],
        "sources": [],
        "answer": "",
        "memory": ensure_memory(memory),
        "memory_used": False,
        "memory_reason": "",
        "memory_snapshot": {},
        "agent_trace": [],
        "skill_trace": [],
        "graph_trace": [],
        "errors": [],
        "created_at": _now(),
    }


def _state_to_result(state: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
    keys = [
        "request_id",
        "agent_engine",
        "question",
        "resolved_question",
        "answer",
        "sources",
        "retrieved_chunks",
        "small_retrieved_chunks",
        "long_context",
        "retrieval_logs",
        "agent_trace",
        "skill_trace",
        "graph_trace",
        "query_type",
        "query_type_label",
        "query_reason",
        "context_sufficient",
        "context_reason",
        "context_coverage",
        "support_level",
        "evidence_score",
        "guard_mode",
        "guard_details",
        "claim_verification",
        "rewritten_queries",
        "rewrite_steps",
        "retriever_mode",
        "top_k",
        "candidate_k",
        "max_rewrites",
        "use_rerank",
        "context_mode",
        "memory_used",
        "memory_reason",
        "memory_snapshot",
        "errors",
        "created_at",
        "finalized_at",
    ]
    result = {key: state.get(key) for key in keys}
    result["latency_ms"] = latency_ms
    return result


def run_langgraph_rag_agent(
    question: str,
    top_k: int = 3,
    candidate_k: int = 10,
    max_rewrites: int = 1,
    use_rerank: bool = True,
    context_mode: str = "small_to_big",
    guard_mode: str = "v2",
    memory: Any = None,
    update_memory: bool = True,
) -> Dict[str, Any]:
    """
    LangGraph version of the RAG Agent workflow.

    It keeps the same public result shape as run_rag_agent while exposing
    graph_trace and skill_trace for interview/demo observability.
    """

    if not LANGGRAPH_AVAILABLE:
        result = run_rag_agent(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
            max_rewrites=max_rewrites,
            use_rerank=use_rerank,
            context_mode=context_mode,
            guard_mode=guard_mode,
            memory=memory,
            update_memory=update_memory,
        )
        result["agent_engine"] = "local_fallback"
        result["fallback_reason"] = (
            "LangGraph 未安装或导入失败，已自动回退到本地状态机 Agent。"
        )
        result["langgraph_import_error"] = LANGGRAPH_IMPORT_ERROR
        result.setdefault("graph_trace", [])
        result.setdefault("skill_trace", [])
        return result

    start = perf_counter()
    graph = get_compiled_graph()
    state = _initial_state(
        question=question,
        top_k=top_k,
        candidate_k=candidate_k,
        max_rewrites=max_rewrites,
        use_rerank=use_rerank,
        context_mode=context_mode,
        guard_mode=guard_mode,
        memory=memory,
    )

    if not update_memory:
        state["memory"] = None

    final_state = graph.invoke(state)
    latency_ms = round((perf_counter() - start) * 1000, 2)
    result = _state_to_result(final_state, latency_ms=latency_ms)
    append_agent_log(result)

    return result
