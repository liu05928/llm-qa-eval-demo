import json
import re
from typing import List, Dict, Any

from vector_store import VectorStore


CHUNKS_FILE = "data/chunks/chunks.json"


def load_chunks(chunks_file: str = CHUNKS_FILE) -> List[Dict[str, Any]]:
    """读取 chunks.json"""
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def simple_tokenize(text: str) -> List[str]:
    """
    简单切词函数。
    不额外引入 jieba，避免新增依赖导致项目跑不通。
    """
    if not text:
        return []

    text = text.lower()

    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)

    tokens = english_tokens + chinese_chars

    stopwords = {
        "的", "了", "是", "在", "和", "与", "或", "也", "有", "中", "对",
        "什", "么", "为", "何", "如", "何", "哪", "些", "一", "个",
        "这", "那", "吗", "呢", "吧"
    }

    return [t for t in tokens if t not in stopwords]


def sparse_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    关键词检索：
    1. 读取 chunks
    2. 计算 query 和 chunk 的关键词重叠数量
    3. 按 sparse_score 排序
    4. 返回 top_k
    """
    chunks = load_chunks()
    query_tokens = set(simple_tokenize(query))

    results = []

    for chunk in chunks:
        content = chunk.get("content", "")
        chunk_tokens = set(simple_tokenize(content))

        overlap = query_tokens.intersection(chunk_tokens)
        sparse_score = len(overlap)

        if sparse_score > 0:
            results.append({
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "content": content,
                "sparse_score": sparse_score,
                "dense_score": 0.0,
                "distance": None,
                "hybrid_score": 0.0,
                "retrieval_type": "sparse"
            })

    results = sorted(results, key=lambda x: x["sparse_score"], reverse=True)
    return results[:top_k]


def dense_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    向量检索：
    调用你现有的 VectorStore.search(query, top_k)。
    Chroma 返回的是 distance，distance 越小越相关。
    这里转成 dense_score，方便后续融合和展示。
    """
    vector_store = VectorStore()
    vector_results = vector_store.search(query=query, top_k=top_k)

    results = []

    for item in vector_results:
        distance = item.get("distance")

        if distance is None:
            dense_score = 0.0
        else:
            dense_score = 1 / (1 + float(distance))

        results.append({
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "content": item.get("content"),
            "distance": distance,
            "dense_score": dense_score,
            "sparse_score": 0,
            "hybrid_score": 0.0,
            "retrieval_type": "dense"
        })

    return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 5,
    k: int = 60,
    dense_weight: float = 0.85,
    sparse_weight: float = 0.15
) -> List[Dict[str, Any]]:
    """
    加权 RRF 排名融合。

    原始 RRF:
    score = 1 / (rank + k)

    当前改进：
    - 向量检索 dense 权重更高；
    - 关键词检索 sparse 权重更低；
    - 避免简单关键词匹配引入过多噪声。
    """

    fused = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item.get("chunk_id")

        if chunk_id not in fused:
            fused[chunk_id] = dict(item)
            fused[chunk_id]["rrf_score"] = 0.0
            fused[chunk_id]["dense_rank"] = None
            fused[chunk_id]["sparse_rank"] = None
            fused[chunk_id]["sparse_score"] = item.get("sparse_score", 0)
            fused[chunk_id]["dense_score"] = item.get("dense_score", 0.0)
            fused[chunk_id]["distance"] = item.get("distance")

        fused[chunk_id]["rrf_score"] += dense_weight * (1 / (k + rank))
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["dense_score"] = item.get("dense_score", 0.0)
        fused[chunk_id]["distance"] = item.get("distance")

    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item.get("chunk_id")

        if chunk_id not in fused:
            fused[chunk_id] = dict(item)
            fused[chunk_id]["rrf_score"] = 0.0
            fused[chunk_id]["dense_rank"] = None
            fused[chunk_id]["sparse_rank"] = None
            fused[chunk_id]["dense_score"] = item.get("dense_score", 0.0)
            fused[chunk_id]["distance"] = item.get("distance")

        fused[chunk_id]["rrf_score"] += sparse_weight * (1 / (k + rank))
        fused[chunk_id]["sparse_rank"] = rank
        fused[chunk_id]["sparse_score"] = item.get("sparse_score", 0)

    final_results = list(fused.values())

    for item in final_results:
        item["hybrid_score"] = item.get("rrf_score", 0.0)
        item["retrieval_type"] = "hybrid"

    final_results = sorted(
        final_results,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return final_results[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Hybrid Search 主函数：
    1. 向量检索 candidate_k 条
    2. 关键词检索 candidate_k 条
    3. RRF 融合
    4. 返回 top_k 条
    """
    dense_results = dense_search(query, top_k=candidate_k)
    sparse_results = sparse_search(query, top_k=candidate_k)

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=top_k
    )


if __name__ == "__main__":
    test_question = "什么是 RAG？"

    print("===== Dense Search 测试 =====")
    dense_results = dense_search(test_question, top_k=3)

    for i, item in enumerate(dense_results, start=1):
        print(f"\nTop {i}")
        print("chunk_id:", item.get("chunk_id"))
        print("source:", item.get("source"))
        print("distance:", item.get("distance"))
        print("dense_score:", item.get("dense_score"))
        print("content:", item.get("content", "")[:120])

    print("\n===== Sparse Search 测试 =====")
    sparse_results = sparse_search(test_question, top_k=3)

    for i, item in enumerate(sparse_results, start=1):
        print(f"\nTop {i}")
        print("chunk_id:", item.get("chunk_id"))
        print("source:", item.get("source"))
        print("sparse_score:", item.get("sparse_score"))
        print("content:", item.get("content", "")[:120])

    print("\n===== Hybrid Search 测试 =====")
    hybrid_results = hybrid_search(test_question, top_k=5, candidate_k=10)

    for i, item in enumerate(hybrid_results, start=1):
        print(f"\nTop {i}")
        print("chunk_id:", item.get("chunk_id"))
        print("source:", item.get("source"))
        print("dense_rank:", item.get("dense_rank"))
        print("sparse_rank:", item.get("sparse_rank"))
        print("hybrid_score:", item.get("hybrid_score"))
        print("content:", item.get("content", "")[:120])