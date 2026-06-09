import json
from pathlib import Path
from datetime import datetime

from rag_logger import save_rag_log
from vector_store import VectorStore
from prompt_templates import build_rag_prompt
from llm_client import call_llm
from hybrid_retriever import dense_search, hybrid_search
from reranker import rerank_chunks


RETRIEVAL_LOG_FILE = Path("logs/retrieval_log.json")
VALID_RETRIEVER_MODES = {"vector", "dense_rerank", "bm25_hybrid"}


def format_context(retrieved_chunks):
    """
    将检索到的 chunks 拼接成 RAG Prompt 里的参考资料。
    """

    context_parts = []

    for chunk in retrieved_chunks:
        source = chunk["source"]
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        context_part = f"""
[来源：{source}，片段：{chunk_id}]
{content}
"""
        context_parts.append(context_part.strip())

    return "\n\n".join(context_parts)


def build_sources(retrieved_chunks):
    """
    从检索结果中提取引用来源。
    """

    sources = []

    for chunk in retrieved_chunks:
        sources.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    return sources


def append_retrieval_log(log_data):
    """
    保存检索日志到 logs/retrieval_log.json。
    """

    RETRIEVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RETRIEVAL_LOG_FILE.exists():
        try:
            with RETRIEVAL_LOG_FILE.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []

    logs.append(log_data)

    with RETRIEVAL_LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def add_scores_for_vector_results(chunks):
    """
    给原始向量检索结果补充分数字段，便于页面展示和日志分析。
    Chroma 返回 distance，distance 越小越相关。
    """

    new_chunks = []

    for chunk in chunks:
        item = dict(chunk)
        distance = item.get("distance")

        if distance is None:
            dense_score = 0.0
        else:
            dense_score = 1 / (1 + float(distance))

        item["dense_score"] = dense_score
        item["bm25_score"] = 0.0
        item["hybrid_score"] = 0.0
        item["rerank_score"] = None
        item["rerank_model"] = None
        item["retrieval_type"] = "vector"

        new_chunks.append(item)

    return new_chunks


def build_chunk_log_item(chunk):
    """提取检索日志和页面展示需要的统一分数字段。"""

    return {
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "distance": chunk.get("distance"),
        "dense_score": chunk.get("dense_score"),
        "bm25_score": chunk.get("bm25_score"),
        "hybrid_score": chunk.get("hybrid_score"),
        "rerank_score": chunk.get("rerank_score"),
        "rerank_model": chunk.get("rerank_model"),
        "retrieval_type": chunk.get("retrieval_type"),
        "dense_rank": chunk.get("dense_rank"),
        "bm25_rank": chunk.get("bm25_rank"),
    }


def build_chunk_log_list(chunks):
    """批量提取检索日志字段。"""

    return [build_chunk_log_item(chunk) for chunk in chunks]


def add_final_rerank_scores(question, final_chunks, use_rerank=True):
    """
    对最终上下文补充模型 Rerank 分数，但不改变 final_chunks 原有顺序。

    作用：
    1. Dense-Preserving 策略仍然保留向量检索兜底结果；
    2. 最终 Top-K 中的 dense 兜底片段也能显示 rerank_score；
    3. 页面和日志中可以看到 BAAI/bge-reranker-v2-m3 的模型打分结果。
    """

    if not use_rerank or not final_chunks:
        return final_chunks

    scored_chunks = rerank_chunks(
        query=question,
        chunks=final_chunks,
        top_k=len(final_chunks),
    )

    score_map = {}

    for chunk in scored_chunks:
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            continue

        score_map[chunk_id] = {
            "rerank_score": chunk.get("rerank_score"),
            "rerank_model": chunk.get("rerank_model"),
        }

    updated_chunks = []

    for chunk in final_chunks:
        new_chunk = dict(chunk)
        chunk_id = new_chunk.get("chunk_id")

        if chunk_id in score_map:
            new_chunk["rerank_score"] = score_map[chunk_id].get("rerank_score")
            new_chunk["rerank_model"] = score_map[chunk_id].get("rerank_model")

        updated_chunks.append(new_chunk)

    return updated_chunks


def retrieve_chunks(
    question: str,
    top_k: int = 3,
    retriever_mode: str = "vector",
    candidate_k: int = 10,
    use_rerank: bool = True,
):
    """
    统一检索入口。

    retriever_mode = "vector"：
        使用原始向量检索。

    retriever_mode = "dense_rerank"：
        使用向量召回 candidate_k 条候选，再进行模型 Rerank。

    retriever_mode = "bm25_hybrid"：
        使用向量召回 + BM25 稀疏召回 + RRF 融合，再进行模型 Rerank。

    dense_rerank 和 bm25_hybrid 都采用 Dense-Preserving 策略：
    先保留 dense top2，再用 rerank 后的候选结果补足最终上下文。
    """

    if retriever_mode not in VALID_RETRIEVER_MODES:
        raise ValueError(
            "retriever_mode 只能是 'vector'、'dense_rerank' 或 'bm25_hybrid'"
        )

    if retriever_mode == "vector":
        vector_store = VectorStore()

        retrieved_chunks = vector_store.search(
            query=question,
            top_k=top_k,
        )

        retrieved_chunks = add_scores_for_vector_results(retrieved_chunks)

        retrieval_log = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "retriever_mode": retriever_mode,
            "candidate_k": top_k,
            "final_top_k": top_k,
            "use_rerank": False,
            "strategy": "vector_only",
            "retrieved_candidates": build_chunk_log_list(retrieved_chunks),
            "final_context": build_chunk_log_list(retrieved_chunks),
        }

        return retrieved_chunks, retrieval_log

    dense_candidates = dense_search(
        query=question,
        top_k=candidate_k,
    )

    dense_base_chunks = dense_candidates[:top_k]

    if retriever_mode == "dense_rerank":
        candidate_chunks = dense_candidates
        strategy = "dense_rerank_no_keyword"
    else:
        candidate_chunks = hybrid_search(
            query=question,
            top_k=candidate_k,
            candidate_k=candidate_k,
            dense_results=dense_candidates,
        )
        strategy = "dense_preserving_bm25_hybrid_rerank"

    if use_rerank:
        reranked_chunks = rerank_chunks(
            query=question,
            chunks=candidate_chunks,
            top_k=candidate_k,
        )
    else:
        reranked_chunks = candidate_chunks

    keep_dense_k = min(2, top_k)
    final_chunks = []
    seen_chunk_ids = set()

    for chunk in dense_base_chunks[:keep_dense_k]:
        final_chunks.append(chunk)
        seen_chunk_ids.add(chunk.get("chunk_id"))

    for chunk in reranked_chunks:
        chunk_id = chunk.get("chunk_id")

        if chunk_id not in seen_chunk_ids:
            final_chunks.append(chunk)
            seen_chunk_ids.add(chunk_id)

        if len(final_chunks) >= top_k:
            break

    final_chunks = add_final_rerank_scores(
        question=question,
        final_chunks=final_chunks,
        use_rerank=use_rerank,
    )

    retrieval_log = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "retriever_mode": retriever_mode,
        "candidate_k": candidate_k,
        "final_top_k": top_k,
        "use_rerank": use_rerank,
        "strategy": strategy,
        "dense_base_chunks": build_chunk_log_list(dense_base_chunks),
        "retrieved_candidates": build_chunk_log_list(candidate_chunks),
        "reranked_candidates": build_chunk_log_list(reranked_chunks),
        "final_context": build_chunk_log_list(final_chunks),
    }

    return final_chunks, retrieval_log


def rag_answer(
    question: str,
    top_k: int = 3,
    retriever_mode: str = "vector",
    candidate_k: int = 10,
    use_rerank: bool = True,
):
    """
    RAG 问答主流程。

    流程：
    1. 根据用户问题检索相关 chunks；
    2. 支持 vector、dense_rerank 和 bm25_hybrid 三种检索模式；
    3. dense_rerank / bm25_hybrid 模式下支持模型 Rerank；
    4. 将 chunks 拼接成 context；
    5. 构造 RAG Prompt；
    6. 调用大模型或 Mock 模型生成回答；
    7. 返回 answer、sources、retrieved_chunks。
    """

    retrieved_chunks, retrieval_log = retrieve_chunks(
        question=question,
        top_k=top_k,
        retriever_mode=retriever_mode,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
    )

    context = format_context(retrieved_chunks)

    rag_prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    answer = call_llm(
        question=rag_prompt,
        mode="education",
    )

    result = {
        "question": question,
        "answer": answer,
        "sources": build_sources(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
        "retriever_mode": retriever_mode,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "use_rerank": use_rerank,
    }

    rag_log_data = {
        "question": question,
        "top_k": top_k,
        "retriever_mode": retriever_mode,
        "candidate_k": candidate_k,
        "use_rerank": use_rerank,
        "answer": answer,
        "sources": result["sources"],
        "retrieved_chunks": [
            build_chunk_log_item(chunk)
            for chunk in retrieved_chunks
        ],
    }

    save_rag_log(rag_log_data)

    retrieval_log["answer_length"] = len(answer)
    retrieval_log["sources"] = result["sources"]
    append_retrieval_log(retrieval_log)

    return result


if __name__ == "__main__":
    question = "什么是 RAG？"

    result = rag_answer(
        question=question,
        top_k=3,
        retriever_mode="bm25_hybrid",
        candidate_k=10,
        use_rerank=True,
    )

    print("用户问题：")
    print(result["question"])

    print("\n检索模式：")
    print(result["retriever_mode"])

    print("\n模型回答：")
    print(result["answer"])

    print("\n引用来源：")
    for source in result["sources"]:
        print(f"- {source['source']} / {source['chunk_id']}")

    print("\n检索片段预览：")
    for chunk in result["retrieved_chunks"]:
        print("-" * 50)
        print(f"chunk_id: {chunk.get('chunk_id')}")
        print(f"source: {chunk.get('source')}")
        print(f"distance: {chunk.get('distance')}")
        print(f"dense_score: {chunk.get('dense_score')}")
        print(f"bm25_score: {chunk.get('bm25_score')}")
        print(f"hybrid_score: {chunk.get('hybrid_score')}")
        print(f"rerank_score: {chunk.get('rerank_score')}")
        print(f"rerank_model: {chunk.get('rerank_model')}")
        print(f"content: {chunk.get('content', '')[:150]}...")
