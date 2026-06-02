from rag_logger import save_rag_log
from vector_store import VectorStore
from prompt_templates import build_rag_prompt
from llm_client import call_llm


def format_context(retrieved_chunks):
    """
    将检索到的 chunks 拼接成 RAG Prompt 里的参考资料。

    输入：
    [
        {
            "chunk_id": "rag_intro_001",
            "source": "rag_intro.md",
            "content": "..."
        }
    ]

    输出：
    [来源：rag_intro.md，片段：rag_intro_001]
    文本内容...
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


def rag_answer(question: str, top_k: int = 3):
    """
    RAG 问答主流程。

    流程：
    1. 根据用户问题检索相关 chunks；
    2. 将 chunks 拼接成 context；
    3. 构造 RAG Prompt；
    4. 调用大模型或 Mock 模型生成回答；
    5. 返回 answer、sources、retrieved_chunks。
    """

    vector_store = VectorStore()

    retrieved_chunks = vector_store.search(
        query=question,
        top_k=top_k,
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
    }
    result = {
    "question": question,
    "answer": answer,
    "sources": build_sources(retrieved_chunks),
    "retrieved_chunks": retrieved_chunks,
    }

    log_data = {
    "question": question,
    "top_k": top_k,
    "answer": answer,
    "sources": result["sources"],
    "retrieved_chunks": [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "distance": chunk.get("distance"),
        }
        for chunk in retrieved_chunks
    ],
    }

    save_rag_log(log_data)

    return result


if __name__ == "__main__":
    question = "什么是 RAG？"

    result = rag_answer(
        question=question,
        top_k=3,
    )

    print("用户问题：")
    print(result["question"])

    print("\n模型回答：")
    print(result["answer"])

    print("\n引用来源：")
    for source in result["sources"]:
        print(f"- {source['source']} / {source['chunk_id']}")

    print("\n检索片段预览：")
    for chunk in result["retrieved_chunks"]:
        print("-" * 50)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source: {chunk['source']}")
        print(f"content: {chunk['content'][:150]}...")