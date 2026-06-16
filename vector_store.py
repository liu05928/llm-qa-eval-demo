import json
from pathlib import Path

import chromadb

from embedding_client import EmbeddingClient


CHUNKS_FILE = Path("data/chunks/chunks.json")
VECTOR_DB_DIR = Path("vector_db")
COLLECTION_NAME = "edu_rag_chunks"


class VectorStore:
    """
    Chroma 向量库封装。

    作用：
    1. 读取 chunks.json；
    2. 将 chunk 文本转换成 embedding；
    3. 存入 Chroma 向量数据库；
    4. 根据用户问题检索最相关的 top-k chunks。
    """

    def __init__(self):
        self.embedding_client = EmbeddingClient()

        # PersistentClient 会把向量库持久化保存到 vector_db/ 目录
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def load_chunks(self, chunks_file: Path = CHUNKS_FILE):
        """
        读取 data/chunks/chunks.json。
        """

        if not chunks_file.exists():
            raise FileNotFoundError(f"chunks 文件不存在: {chunks_file}")

        with chunks_file.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        return chunks

    def build_index(self):
        """
        构建向量索引。

        流程：
        1. 读取 chunks.json；
        2. 提取 content；
        3. 生成 embeddings；
        4. 写入 Chroma collection。
        """

        chunks = self.load_chunks()

        if not chunks:
            raise ValueError("chunks.json 为空，无法构建向量索引")

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk.get("chunk_type", "small"),
                "parent_chunk_id": chunk.get("parent_chunk_id", ""),
                "parent_index": int(chunk.get("parent_index") or 0),
                "small_index": int(chunk.get("small_index") or 0),
            }
            for chunk in chunks
        ]

        print(f"准备写入 chunk 数量：{len(chunks)}")

        embeddings = self.embedding_client.get_embeddings(documents)

        # 为了避免重复 add 同样的 id，这里先删除 collection 中已有数据
        existing = self.collection.get()

        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
            print(f"已清空旧索引数量：{len(existing['ids'])}")

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"向量索引构建完成，写入数量：{len(ids)}")

    def search(self, query: str, top_k: int = 3):
        """
        根据用户问题检索最相关的 top-k chunks。

        返回格式：
        [
            {
                "chunk_id": "...",
                "source": "...",
                "content": "...",
                "distance": 0.123
            }
        ]
        """

        query_embedding = self.embedding_client.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            item = dict(metadata)
            item["chunk_id"] = metadata.get("chunk_id") or chunk_id
            item["source"] = metadata.get("source")
            item["content"] = document
            item["distance"] = distance
            retrieved_chunks.append(item)

        return retrieved_chunks


if __name__ == "__main__":
    vector_store = VectorStore()

    print("开始构建向量索引...")
    vector_store.build_index()

    test_questions = [
        "什么是 RAG？",
        "Agent 和普通聊天机器人有什么区别？",
        "Prompt Engineering 有什么作用？",
        "大模型在教育场景中可以做什么？",
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"问题：{question}")

        results = vector_store.search(question, top_k=3)

        for i, item in enumerate(results, start=1):
            print("-" * 50)
            print(f"Top {i}")
            print(f"chunk_id: {item['chunk_id']}")
            print(f"source: {item['source']}")
            print(f"distance: {item['distance']}")
            print(f"content: {item['content'][:150]}...")
