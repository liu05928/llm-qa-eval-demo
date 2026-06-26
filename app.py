from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from agent_session_store import (
    get_or_create_session,
    get_session_snapshot,
    reset_session,
    save_session,
)
from rag_agent import run_rag_agent
from langgraph_agent import run_langgraph_rag_agent
from rag_pipeline import rag_answer
from runtime_controls import (
    build_cache_key,
    check_rate_limit,
    get_cached_response,
    get_runtime_controls_status,
    set_cached_response,
)

from config import MODEL, USE_MOCK
from llm_client import call_llm
from chat_logger import save_log
from prompt_templates import get_available_modes
from evaluator import run_evaluation, load_eval_results


app = FastAPI(
    title="企业知识库 RAG Agent 问答与检索优化系统",
    description="面向企业内部文档问答场景的 RAG Agent 服务，支持 LangGraph/本地状态机双引擎、会话记忆、BM25 Hybrid、Rerank、Small-to-Big Long-text RAG、来源引用、缓存限流和自动评测。",
    version="0.8.0"
)


class ChatRequest(BaseModel):
    """
    用户请求体。

    question: 用户问题
    mode: 问答模式
    """

    question: str
    mode: Literal["general", "education", "paper_summary"] = "general"

class RagChatRequest(BaseModel):
    """
    RAG 知识库问答请求体。

    question: 用户问题
    top_k: 最终用于回答的文本块数量
    retriever_mode: 检索模式，支持 vector、dense_rerank、bm25_hybrid
    context_mode: 上下文模式，small 表示小 chunk 直接回答，small_to_big 表示小 chunk 召回后扩展父段落
    candidate_k: dense_rerank / bm25_hybrid 模式下候选召回数量
    use_rerank: 是否启用 Rerank 重排序
    """

    question: str
    top_k: int = 3
    retriever_mode: Literal["vector", "dense_rerank", "bm25_hybrid"] = "vector"
    context_mode: Literal["small", "small_to_big"] = "small_to_big"
    candidate_k: int = 10
    use_rerank: bool = True


class AgentChatRequest(BaseModel):
    """
    Agent 知识库问答请求体。

    session_id: 会话 ID，为空时自动生成
    context_mode: 上下文模式，默认 small_to_big
    reset_memory: 是否在本轮问答前清空当前 session 记忆
    """

    question: str
    session_id: Optional[str] = None
    agent_engine: Literal["langgraph", "local"] = "langgraph"
    top_k: int = 3
    candidate_k: int = 10
    max_rewrites: int = 1
    use_rerank: bool = True
    context_mode: Literal["small", "small_to_big"] = "small_to_big"
    reset_memory: bool = False
    enable_cache: bool = False
    enable_rate_limit: bool = True

class ChatResponse(BaseModel):
    """
    系统返回体。
    """

    question: str
    answer: str
    mode: str
    model: str
    mock: bool


@app.get("/")
def root():
    """
    根路径接口，用来检查服务是否启动成功。
    """

    return {
        "message": "企业知识库 RAG Agent 问答接口系统已启动",
        "docs": "请访问 /docs 查看接口文档",
        "available_modes": get_available_modes()
    }


@app.get("/health")
def health_check():
    """
    健康检查接口。
    """

    return {
        "status": "ok",
        "model": MODEL,
        "mock": USE_MOCK,
        "available_modes": get_available_modes(),
        "agent_api": {
            "chat": "/agent/chat",
            "session": "/agent/session/{session_id}",
            "engines": ["langgraph", "local"],
        },
        "runtime_controls": get_runtime_controls_status(),
    }


@app.get("/modes")
def list_modes():
    """
    查看当前支持的所有问答模式。
    """

    return {
        "available_modes": get_available_modes(),
        "description": {
            "general": "通用问答模式",
            "education": "教育解释模式",
            "paper_summary": "论文总结模式"
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    问答接口。

    输入示例：
    {
        "question": "什么是 RAG？",
        "mode": "education"
    }
    """

    question = request.question.strip()
    mode = request.mode

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        answer = call_llm(question, mode=mode)
        save_log(question, answer, mode=mode)

        return ChatResponse(
            question=question,
            answer=answer,
            mode=mode,
            model=MODEL,
            mock=USE_MOCK
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_chat")
def rag_chat(request: RagChatRequest):
    """
    RAG 知识库问答接口。

    支持三种检索模式：
    1. vector：基础向量检索；
    2. dense_rerank：向量召回 + Rerank，无关键词/BM25 召回；
    3. bm25_hybrid：向量召回 + BM25 召回 + RRF 融合 + Rerank。

    输入：
    - question: 用户问题
    - top_k: 最终用于回答的文本块数量
    - retriever_mode: 检索模式，vector、dense_rerank 或 bm25_hybrid
    - candidate_k: dense_rerank / bm25_hybrid 模式下候选召回数量
    - use_rerank: 是否启用 Rerank 重排序
    - context_mode: small 或 small_to_big

    输出：
    - question: 原始问题
    - answer: 模型回答
    - sources: 引用来源
    - retrieved_chunks: 检索到的文本块
    - retriever_mode: 当前使用的检索模式
    - candidate_k: 候选召回数量
    - use_rerank: 是否启用 Rerank
    - context_mode: 当前上下文模式
    - small_retrieved_chunks: 用于召回的小 chunk
    - long_context: Small-to-Big 扩展摘要
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = rag_answer(
            question=question,
            top_k=request.top_k,
            retriever_mode=request.retriever_mode,
            candidate_k=request.candidate_k,
            use_rerank=request.use_rerank,
            context_mode=request.context_mode,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat")
def agent_chat(request: AgentChatRequest, http_request: Request):
    """
    支持会话记忆的 Agent 问答接口。

    - 不传 session_id 时自动生成；
    - 传入 session_id 时复用对应会话记忆；
    - reset_memory=true 时先清空该 session 的短期记忆；
    - agent_engine=langgraph 时使用 LangGraph Skills 编排，local 时使用本地状态机；
    - enable_cache=true 时仅缓存无显式 session 的热点单轮问题，避免污染多轮记忆；
    - enable_rate_limit=true 时在 Redis 可用时进行固定窗口限流；
    - 返回 resolved_question、memory_snapshot 和 agent_trace，便于调试。
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        session_id, memory, session_created = get_or_create_session(request.session_id)

        if request.reset_memory:
            reset_session(session_id)
            session_id, memory, _ = get_or_create_session(session_id)

        rate_limit_info = {
            "enabled": False,
            "allowed": True,
            "reason": "disabled by request",
        }

        if request.enable_rate_limit:
            client_host = http_request.client.host if http_request.client else "unknown"
            rate_identifier = session_id or client_host
            allowed, rate_limit_info = check_rate_limit(rate_identifier)

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "请求过于频繁，请稍后再试。",
                        "rate_limit": rate_limit_info,
                    },
                )

        cache_allowed = (
            request.enable_cache
            and not request.session_id
            and not request.reset_memory
        )
        cache_status = "disabled"
        cache_key = ""

        if request.enable_cache and not cache_allowed:
            cache_status = "skipped_for_session_memory"

        if cache_allowed and not get_runtime_controls_status().get("redis_available"):
            cache_status = "unavailable"
            cache_allowed = False

        if cache_allowed:
            cache_key = build_cache_key(
                namespace="agent_chat",
                payload={
                    "question": question,
                    "agent_engine": request.agent_engine,
                    "top_k": request.top_k,
                    "candidate_k": request.candidate_k,
                    "max_rewrites": request.max_rewrites,
                    "use_rerank": request.use_rerank,
                    "context_mode": request.context_mode,
                    "memory_snapshot": memory.to_dict(),
                },
            )
            cached_result = get_cached_response(cache_key)

            if cached_result:
                memory.update_from_result(
                    question=question,
                    resolved_question=cached_result.get("resolved_question", question),
                    answer=cached_result.get("answer", ""),
                    sources=cached_result.get("sources", []),
                    query_type=cached_result.get("query_type", "general"),
                    retriever_mode=cached_result.get("retriever_mode", ""),
                    rewritten_queries=cached_result.get("rewritten_queries", []),
                )
                save_session(session_id, memory)

                cached_result["session_id"] = session_id
                cached_result["session_created"] = session_created
                cached_result["reset_memory"] = request.reset_memory
                cached_result["cache_hit"] = True
                cached_result["cache_status"] = "hit"
                cached_result["cache_key"] = cache_key
                cached_result["rate_limit"] = rate_limit_info
                cached_result["memory_snapshot"] = memory.to_dict()
                return cached_result

            cache_status = "miss"

        if request.agent_engine == "langgraph":
            result = run_langgraph_rag_agent(
                question=question,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                max_rewrites=request.max_rewrites,
                use_rerank=request.use_rerank,
                context_mode=request.context_mode,
                memory=memory,
            )
        else:
            result = run_rag_agent(
                question=question,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                max_rewrites=request.max_rewrites,
                use_rerank=request.use_rerank,
                context_mode=request.context_mode,
                memory=memory,
            )
            result["agent_engine"] = "local"

        save_session(session_id, memory)

        result["session_id"] = session_id
        result["session_created"] = session_created
        result["reset_memory"] = request.reset_memory
        result["cache_hit"] = False
        result["cache_status"] = cache_status
        result["cache_key"] = cache_key
        result["rate_limit"] = rate_limit_info

        if cache_allowed and cache_key:
            cached_payload = dict(result)
            cached_payload.pop("session_id", None)
            cached_payload.pop("session_created", None)
            cached_payload.pop("reset_memory", None)
            cached_payload.pop("cache_hit", None)
            cached_payload.pop("cache_status", None)
            cached_payload.pop("cache_key", None)
            cached_payload.pop("rate_limit", None)
            set_cached_response(cache_key, cached_payload)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/session/{session_id}")
def get_agent_session(session_id: str):
    """
    查看指定 Agent session 的会话记忆。
    """

    try:
        return get_session_snapshot(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session 不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/agent/session/{session_id}")
def delete_agent_session(session_id: str):
    """
    清空指定 Agent session 的会话记忆。
    """

    try:
        session = reset_session(session_id)
        return {
            "session_id": session_id,
            "status": "reset",
            "memory": session.get("memory", {}),
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="session 不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
def evaluate():
    """
    运行一轮自动评测。

    系统会读取 data/test_questions.json，
    对每个测试问题调用 call_llm，
    然后根据 expected_keywords 计算关键词命中得分。
    """

    try:
        summary = run_evaluation()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval-results")
def get_eval_results():
    """
    查看最近一次评测结果。
    """

    return load_eval_results()
