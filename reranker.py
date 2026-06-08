from typing import List, Dict, Any

from hybrid_retriever import simple_tokenize, hybrid_search


def calc_keyword_overlap_score(query: str, content: str) -> float:
    """
    计算问题和 chunk 内容的关键词重叠得分。
    """
    query_tokens = set(simple_tokenize(query))
    content_tokens = set(simple_tokenize(content))

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(content_tokens)
    return len(overlap) / len(query_tokens)


def calc_length_score(content: str) -> float:
    """
    简单长度得分。
    太短的 chunk 信息不足，太长的 chunk 可能噪声较多。
    """
    length = len(content)

    if 100 <= length <= 800:
        return 1.0
    elif 50 <= length < 100:
        return 0.7
    elif 800 < length <= 1200:
        return 0.7
    else:
        return 0.4


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    对候选 chunks 进行轻量级重排序。

    改进策略：
    1. 更重视 dense_score，保证语义检索结果不被简单关键词检索挤掉；
    2. 保留 hybrid_score，体现融合检索结果；
    3. 降低 sparse_score 和 keyword_overlap 的影响，减少关键词噪声。
    """

    reranked = []

    for chunk in chunks:
        content = chunk.get("content", "")

        hybrid_score = float(chunk.get("hybrid_score", 0.0) or 0.0)
        dense_score = float(chunk.get("dense_score", 0.0) or 0.0)
        sparse_score = float(chunk.get("sparse_score", 0.0) or 0.0)

        keyword_overlap_score = calc_keyword_overlap_score(query, content)
        length_score = calc_length_score(content)

        dense_rank = chunk.get("dense_rank")
        sparse_rank = chunk.get("sparse_rank")

        if dense_rank is None:
            dense_rank_score = 0.0
        else:
            dense_rank_score = 1 / dense_rank

        if sparse_rank is None:
            sparse_rank_score = 0.0
        else:
            sparse_rank_score = 1 / sparse_rank

        rerank_score = (
            0.40 * dense_score
            + 0.25 * dense_rank_score
            + 0.20 * hybrid_score
            + 0.07 * keyword_overlap_score
            + 0.04 * min(sparse_score / 10, 1.0)
            + 0.02 * sparse_rank_score
            + 0.02 * length_score
        )

        new_chunk = dict(chunk)
        new_chunk["keyword_overlap_score"] = keyword_overlap_score
        new_chunk["length_score"] = length_score
        new_chunk["rerank_score"] = rerank_score

        reranked.append(new_chunk)

    reranked = sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]

if __name__ == "__main__":
    test_question = "什么是 RAG？"

    candidates = hybrid_search(
        query=test_question,
        top_k=10,
        candidate_k=10
    )

    final_chunks = rerank_chunks(
        query=test_question,
        chunks=candidates,
        top_k=3
    )

    print("===== Rerank 测试 =====")

    for i, item in enumerate(final_chunks, start=1):
        print(f"\nTop {i}")
        print("chunk_id:", item.get("chunk_id"))
        print("source:", item.get("source"))
        print("hybrid_score:", item.get("hybrid_score"))
        print("keyword_overlap_score:", item.get("keyword_overlap_score"))
        print("rerank_score:", item.get("rerank_score"))
        print("content:", item.get("content", "")[:150])