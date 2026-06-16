import hashlib
import json
import re
from pathlib import Path

from document_loader import load_documents


CHUNKS_DIR = Path("data/chunks")
CHUNKS_FILE = CHUNKS_DIR / "chunks.json"
BIG_CHUNKS_FILE = CHUNKS_DIR / "big_chunks.json"

DEFAULT_SMALL_CHUNK_SIZE = 500
DEFAULT_SMALL_CHUNK_OVERLAP = 80
DEFAULT_BIG_CHUNK_SIZE = 1600
DEFAULT_BIG_CHUNK_OVERLAP = 180


def build_safe_stem(source: str, max_length: int = 80) -> str:
    stem = Path(source).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", stem).strip("_")

    if not safe_stem:
        safe_stem = "chunk"

    return safe_stem[:max_length].rstrip("_")


def build_chunk_id(source: str, index: int) -> str:
    """
    根据来源路径和序号生成稳定 chunk_id。

    例如：
    rag_intro.md 的第 1 个文本块 → rag_intro_001
    science_textbooks/foo.md 的第 1 个文本块 → science_textbooks_foo_xxxxxxxx_001
    """

    safe_stem = build_safe_stem(source)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]

    return f"{safe_stem}_{digest}_{index:03d}"


def build_big_chunk_id(source: str, index: int) -> str:
    safe_stem = build_safe_stem(source)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}_big_{index:03d}"


def build_small_chunk_id(parent_chunk_id: str, index: int) -> str:
    return f"{parent_chunk_id}_small_{index:03d}"


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


def split_text_to_big_chunks(
    text: str,
    big_chunk_size: int = DEFAULT_BIG_CHUNK_SIZE,
    big_chunk_overlap: int = DEFAULT_BIG_CHUNK_OVERLAP,
):
    """
    按 Markdown 段落优先切分大 chunk。

    Small-to-Big RAG 中，大 chunk 不是用于召回，而是用于回答阶段提供更完整上下文。
    因此这里尽量保持标题、段落和表格附近内容在同一个父级片段里。
    """

    if big_chunk_size <= 0:
        raise ValueError("big_chunk_size 必须大于 0")

    if big_chunk_overlap < 0:
        raise ValueError("big_chunk_overlap 不能小于 0")

    if big_chunk_overlap >= big_chunk_size:
        raise ValueError("big_chunk_overlap 必须小于 big_chunk_size")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]

    if not paragraphs:
        return []

    big_chunks = []
    current_parts = []
    current_length = 0

    def flush_current():
        nonlocal current_parts, current_length

        if not current_parts:
            return

        big_chunks.append("\n\n".join(current_parts).strip())

        overlap_parts = []
        overlap_length = 0

        for part in reversed(current_parts):
            part_length = len(part)

            if overlap_parts and overlap_length + part_length > big_chunk_overlap:
                break

            overlap_parts.insert(0, part)
            overlap_length += part_length

        current_parts = overlap_parts
        current_length = sum(len(part) for part in current_parts)

    for paragraph in paragraphs:
        if len(paragraph) > big_chunk_size:
            flush_current()
            big_chunks.extend(
                split_text(
                    paragraph,
                    chunk_size=big_chunk_size,
                    chunk_overlap=big_chunk_overlap,
                )
            )
            current_parts = []
            current_length = 0
            continue

        next_length = current_length + len(paragraph)

        if current_parts:
            next_length += 2

        if current_parts and next_length > big_chunk_size:
            flush_current()

        current_parts.append(paragraph)
        current_length += len(paragraph)

    flush_current()

    return big_chunks


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

    small_chunks, _ = split_documents_small_to_big(
        documents=documents,
        small_chunk_size=chunk_size,
        small_chunk_overlap=chunk_overlap,
    )

    return small_chunks


def split_documents_small_to_big(
    documents,
    small_chunk_size: int = DEFAULT_SMALL_CHUNK_SIZE,
    small_chunk_overlap: int = DEFAULT_SMALL_CHUNK_OVERLAP,
    big_chunk_size: int = DEFAULT_BIG_CHUNK_SIZE,
    big_chunk_overlap: int = DEFAULT_BIG_CHUNK_OVERLAP,
):
    """
    生成 Small-to-Big RAG 需要的两级 chunk。

    - small_chunks: 小 chunk，用于向量/BM25 召回；
    - big_chunks: 父级大段落，用于回答阶段扩展上下文。
    """

    all_small_chunks = []
    all_big_chunks = []

    for doc in documents:
        source = doc["source"]
        content = doc["content"]

        parent_text_chunks = split_text_to_big_chunks(
            text=content,
            big_chunk_size=big_chunk_size,
            big_chunk_overlap=big_chunk_overlap,
        )

        for parent_index, parent_content in enumerate(parent_text_chunks, start=1):
            parent_chunk_id = build_big_chunk_id(source, parent_index)
            child_chunk_ids = []
            small_text_chunks = split_text(
                text=parent_content,
                chunk_size=small_chunk_size,
                chunk_overlap=small_chunk_overlap,
            )

            for small_index, chunk_content in enumerate(small_text_chunks, start=1):
                chunk_id = build_small_chunk_id(parent_chunk_id, small_index)
                child_chunk_ids.append(chunk_id)
                chunk = {
                    "chunk_id": chunk_id,
                    "source": source,
                    "content": chunk_content,
                    "chunk_type": "small",
                    "parent_chunk_id": parent_chunk_id,
                    "parent_index": parent_index,
                    "small_index": small_index,
                }

                all_small_chunks.append(chunk)

            parent_chunk = {
                "chunk_id": parent_chunk_id,
                "source": source,
                "content": parent_content,
                "chunk_type": "big",
                "parent_chunk_id": parent_chunk_id,
                "parent_index": parent_index,
                "child_chunk_ids": child_chunk_ids,
                "child_count": len(child_chunk_ids),
                "char_count": len(parent_content),
            }

            all_big_chunks.append(parent_chunk)

    return all_small_chunks, all_big_chunks


def save_chunks(chunks, chunks_file: Path = CHUNKS_FILE):
    """
    将 chunks 保存为 JSON 文件。
    """

    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    with chunks_file.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def save_big_chunks(big_chunks, big_chunks_file: Path = BIG_CHUNKS_FILE):
    """
    将父级大 chunk 保存为 JSON 文件。
    """

    big_chunks_file.parent.mkdir(parents=True, exist_ok=True)

    with big_chunks_file.open("w", encoding="utf-8") as f:
        json.dump(big_chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    documents = load_documents()

    chunks, big_chunks = split_documents_small_to_big(
        documents=documents,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
        small_chunk_overlap=DEFAULT_SMALL_CHUNK_OVERLAP,
        big_chunk_size=DEFAULT_BIG_CHUNK_SIZE,
        big_chunk_overlap=DEFAULT_BIG_CHUNK_OVERLAP,
    )

    save_chunks(chunks)
    save_big_chunks(big_chunks)

    print(f"成功读取文档数：{len(documents)}")
    print(f"成功生成 small chunk 数：{len(chunks)}")
    print(f"成功生成 big chunk 数：{len(big_chunks)}")
    print(f"chunks 已保存到：{CHUNKS_FILE}")
    print(f"big_chunks 已保存到：{BIG_CHUNKS_FILE}")

    print("\n前 3 个 chunk 预览：")

    for chunk in chunks[:3]:
        print("-" * 50)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source: {chunk['source']}")
        print(f"content: {chunk['content'][:120]}...")
