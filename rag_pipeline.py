import json
from pathlib import Path
from datetime import datetime

from rag_logger import save_rag_log
from vector_store import VectorStore
from prompt_templates import build_rag_prompt
from llm_client import call_llm
from hybrid_retriever import hybrid_search
from reranker import rerank_chunks


RETRIEVAL_LOG_FILE = Path("logs/retrieval_log.json")


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
        item["sparse_score"] = 0
        item["hybrid_score"] = 0.0
        item["rerank_score"] = None
        item["retrieval_type"] = "vector"

        new_chunks.append(item)

    return new_chunks


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

    retriever_mode = "hybrid"：
        使用 Dense-Preserving Hybrid Search + Rerank。

        设计思路：
        1. 先保留基础向量检索结果，保证语义检索稳定性；
        2. 再使用 Hybrid Search 扩大候选召回范围；
        3. 对 Hybrid 候选结果进行 Rerank；
        4. 最终结果中优先保留 dense top2，剩余位置由 rerank 后的 hybrid 结果补充。
    """

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
            "retrieved_candidates": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in retrieved_chunks
            ],
            "final_context": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in retrieved_chunks
            ],
        }

        return retrieved_chunks, retrieval_log

    if retriever_mode == "hybrid":
        # 1. 基础向量检索结果作为稳定兜底
        vector_store = VectorStore()

        dense_base_chunks = vector_store.search(
            query=question,
            top_k=top_k,
        )

        dense_base_chunks = add_scores_for_vector_results(dense_base_chunks)

        # 2. Hybrid Search 扩大候选召回范围
        candidate_chunks = hybrid_search(
            query=question,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        # 3. 对 Hybrid 候选结果进行 Rerank
        if use_rerank:
            reranked_chunks = rerank_chunks(
                query=question,
                chunks=candidate_chunks,
                top_k=candidate_k,
            )
        else:
            reranked_chunks = candidate_chunks

        # 4. Dense-Preserving 安全融合
        #    默认至少保留 dense top2，避免简单关键词检索带来噪声
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

        retrieval_log = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "retriever_mode": retriever_mode,
            "candidate_k": candidate_k,
            "final_top_k": top_k,
            "use_rerank": use_rerank,
            "strategy": "dense_preserving_hybrid_rerank",
            "dense_base_chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in dense_base_chunks
            ],
            "retrieved_candidates": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in candidate_chunks
            ],
            "reranked_candidates": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in reranked_chunks
            ],
            "final_context": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "distance": chunk.get("distance"),
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "hybrid_score": chunk.get("hybrid_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in final_chunks
            ],
        }

        return final_chunks, retrieval_log

    raise ValueError("retriever_mode 只能是 'vector' 或 'hybrid'")


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
    2. 支持 vector 和 hybrid 两种检索模式；
    3. hybrid 模式下支持 Rerank；
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
            {
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "distance": chunk.get("distance"),
                "dense_score": chunk.get("dense_score"),
                "sparse_score": chunk.get("sparse_score"),
                "hybrid_score": chunk.get("hybrid_score"),
                "rerank_score": chunk.get("rerank_score"),
            }
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
        retriever_mode="hybrid",
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
        print(f"sparse_score: {chunk.get('sparse_score')}")
        print(f"hybrid_score: {chunk.get('hybrid_score')}")
        print(f"rerank_score: {chunk.get('rerank_score')}")
        print(f"content: {chunk.get('content', '')[:150]}...")