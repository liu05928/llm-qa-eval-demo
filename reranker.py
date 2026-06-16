from typing import List, Dict, Any

import requests

from config import (
    USE_MOCK,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    RERANK_MODEL,
)
from hybrid_retriever import hybrid_search


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


class RerankerClient:
    """
    Reranker 客户端。

    USE_MOCK=false:
        调用硅基流动 BAAI/bge-reranker-v2-m3 进行模型重排序。

    USE_MOCK=true:
        使用简单规则 mock 分数，方便本地跑通流程。
    """

    def __init__(self):
        self.model = RERANK_MODEL
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.rerank_url = f"{self.base_url}/rerank"

    def _mock_score(self, query: str, document: str) -> float:
        """
        Mock Rerank 分数。
        只用于 USE_MOCK=true 时跑通流程。
        """
        query_chars = set(query)
        doc_chars = set(document)

        if not query_chars:
            return 0.0

        return len(query_chars.intersection(doc_chars)) / len(query_chars)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        调用 Rerank API。

        返回格式示例：
        [
            {
                "index": 0,
                "relevance_score": 0.98
            }
        ]
        """

        if not documents:
            return []

        if USE_MOCK:
            scored = []

            for idx, doc in enumerate(documents):
                scored.append(
                    {
                        "index": idx,
                        "relevance_score": self._mock_score(query, doc),
                    }
                )

            scored = sorted(
                scored,
                key=lambda x: x["relevance_score"],
                reverse=True,
            )

            return scored[:top_n]

        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY 未配置，无法调用硅基流动 Rerank API")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

        response = requests.post(
            self.rerank_url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Rerank API 调用失败，状态码：{response.status_code}，返回：{response.text}"
            )

        data = response.json()

        return data.get("results", [])


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    使用硅基流动 Rerank 模型对候选 chunks 进行重排序。
    """

    if not chunks:
        return []

    documents = [chunk.get("content", "") for chunk in chunks]

    client = RerankerClient()

    rerank_results = client.rerank(
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
    )

    final_chunks = []
    max_bm25_score = max(
        [_safe_float(chunk.get("bm25_score")) for chunk in chunks],
        default=0.0,
    )
    max_hybrid_score = max(
        [_safe_float(chunk.get("hybrid_score")) for chunk in chunks],
        default=0.0,
    )

    for item in rerank_results:
        index = item.get("index")
        score = item.get("relevance_score", item.get("score", 0.0))

        if index is None:
            continue

        chunk = dict(chunks[index])
        bm25_norm = (
            _safe_float(chunk.get("bm25_score")) / max_bm25_score
            if max_bm25_score > 0
            else 0.0
        )
        hybrid_norm = (
            _safe_float(chunk.get("hybrid_score")) / max_hybrid_score
            if max_hybrid_score > 0
            else 0.0
        )
        if max_bm25_score > 0 and _safe_float(chunk.get("bm25_score")) > 0:
            retrieval_score = 0.8 * bm25_norm + 0.2 * hybrid_norm
        else:
            retrieval_score = hybrid_norm

        combined_score = 0.6 * _safe_float(score) + 0.4 * retrieval_score

        chunk["rerank_score"] = score
        chunk["rerank_combined_score"] = round(combined_score, 6)
        chunk["rerank_model"] = client.model
        chunk["retrieval_type"] = "model_rerank"

        final_chunks.append(chunk)

    final_chunks = sorted(
        final_chunks,
        key=lambda chunk: chunk.get("rerank_combined_score", 0.0),
        reverse=True,
    )

    return final_chunks[:top_k]


if __name__ == "__main__":
    test_question = "什么是 RAG？"

    candidates = hybrid_search(
        query=test_question,
        top_k=10,
        candidate_k=10,
    )

    final_chunks = rerank_chunks(
        query=test_question,
        chunks=candidates,
        top_k=3,
    )

    print("Rerank 模型：", RERANK_MODEL)
    print("===== 模型 Rerank 测试 =====")

    for i, item in enumerate(final_chunks, start=1):
        print(f"\nTop {i}")
        print("chunk_id:", item.get("chunk_id"))
        print("source:", item.get("source"))
        print("hybrid_score:", item.get("hybrid_score"))
        print("rerank_score:", item.get("rerank_score"))
        print("rerank_model:", item.get("rerank_model"))
        print("content:", item.get("content", "")[:150])
