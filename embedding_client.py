from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingClient:
    """
    文本向量化客户端。

    作用：
    1. 加载 sentence-transformers 模型；
    2. 把单条文本转换成 embedding 向量；
    3. 把多条文本批量转换成 embedding 向量。
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        print(f"正在加载 embedding 模型：{self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("embedding 模型加载完成")

    def get_embedding(self, text: str):
        """
        将单条文本转换为向量。
        """
        embedding = self.model.encode(text)

        # Chroma 需要普通 Python list，不能直接用 numpy array
        return embedding.tolist()

    def get_embeddings(self, texts: list[str]):
        """
        将多条文本批量转换为向量。
        """
        embeddings = self.model.encode(texts)

        return embeddings.tolist()


if __name__ == "__main__":
    client = EmbeddingClient()

    text = "RAG 是检索增强生成，可以结合外部知识库回答问题。"

    embedding = client.get_embedding(text)

    print(f"文本：{text}")
    print(f"向量维度：{len(embedding)}")
    print(f"前 5 个向量值：{embedding[:5]}")