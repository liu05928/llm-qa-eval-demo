import json
from pathlib import Path

from document_loader import load_documents


CHUNKS_DIR = Path("data/chunks")
CHUNKS_FILE = CHUNKS_DIR / "chunks.json"


def build_chunk_id(source: str, index: int) -> str:
    """
    根据文件名和序号生成 chunk_id。

    例如：
    rag_intro.md 的第 1 个文本块 → rag_intro_001
    agent_intro.md 的第 2 个文本块 → agent_intro_002
    """

    stem = Path(source).stem
    return f"{stem}_{index:03d}"


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80):
    """
    把一整段长文本切分成多个 chunk。

    参数：
    - chunk_size: 每个文本块的最大字符数
    - chunk_overlap: 相邻文本块之间重叠的字符数

    返回：
    [
        "第一个文本块",
        "第二个文本块",
        ...
    ]
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 下一段从当前 end 往前重叠 chunk_overlap 个字符的位置开始
        start = end - chunk_overlap

    return chunks


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 80):
    """
    把多个文档切分成 chunks。

    输入：
    [
        {
            "source": "rag_intro.md",
            "content": "文档正文..."
        }
    ]

    输出：
    [
        {
            "chunk_id": "rag_intro_001",
            "source": "rag_intro.md",
            "content": "文本块内容..."
        }
    ]
    """

    all_chunks = []

    for doc in documents:
        source = doc["source"]
        content = doc["content"]

        text_chunks = split_text(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk_content in enumerate(text_chunks, start=1):
            chunk = {
                "chunk_id": build_chunk_id(source, index),
                "source": source,
                "content": chunk_content,
            }

            all_chunks.append(chunk)

    return all_chunks


def save_chunks(chunks, chunks_file: Path = CHUNKS_FILE):
    """
    将 chunks 保存为 JSON 文件。
    """

    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    with chunks_file.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    documents = load_documents()

    chunks = split_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=80,
    )

    save_chunks(chunks)

    print(f"成功读取文档数：{len(documents)}")
    print(f"成功生成 chunk 数：{len(chunks)}")
    print(f"chunks 已保存到：{CHUNKS_FILE}")

    print("\n前 3 个 chunk 预览：")

    for chunk in chunks[:3]:
        print("-" * 50)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source: {chunk['source']}")
        print(f"content: {chunk['content'][:120]}...")