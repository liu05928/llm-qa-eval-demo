from typing import List

import requests

from config import (
    USE_MOCK,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    EMBEDDING_MODEL,
)


class EmbeddingClient:
    """
    Embedding 客户端。

    USE_MOCK=false:
        调用硅基流动 BAAI/bge-m3 生成向量。

    USE_MOCK=true:
        使用本地简单 mock 向量，方便无 API Key 时测试流程。
        注意：mock 向量只用于跑通流程，不代表真实检索效果。
    """

    def __init__(self):
        self.model = EMBEDDING_MODEL
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL.rstrip("/")
        self.embedding_url = f"{self.base_url}/embeddings"

    def _mock_embedding(self, text: str, dim: int = 1024) -> List[float]:
        """
        简单 mock embedding。
        只用于 USE_MOCK=true 时跑通流程。
        """
        values = [0.0] * dim

        if not text:
            return values

        for i, ch in enumerate(text[:dim]):
            values[i % dim] += (ord(ch) % 100) / 100.0

        norm = sum(v * v for v in values) ** 0.5

        if norm == 0:
            return values

        return [v / norm for v in values]

    def get_embedding(self, text: str) -> List[float]:
        """
        单条文本转向量。
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本转向量。
        """
        if not texts:
            return []

        if USE_MOCK:
            return [self._mock_embedding(text) for text in texts]

        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY 未配置，无法调用硅基流动 Embedding API")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(
            self.embedding_url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding API 调用失败，状态码：{response.status_code}，返回：{response.text}"
            )

        data = response.json()

        embeddings = [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]

        return embeddings


if __name__ == "__main__":
    client = EmbeddingClient()

    texts = [
        "什么是 RAG？",
        "RAG 是检索增强生成。",
    ]

    embeddings = client.get_embeddings(texts)

    print("Embedding 模型：", client.model)
    print("向量数量：", len(embeddings))
    print("向量维度：", len(embeddings[0]) if embeddings else 0)
    print("前 5 维：", embeddings[0][:5])