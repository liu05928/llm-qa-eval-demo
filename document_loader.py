from pathlib import Path


RAW_DOCS_DIR = Path("data/raw_docs")


def load_documents(raw_docs_dir: Path = RAW_DOCS_DIR):
    """
    读取 data/raw_docs/ 目录下的 Markdown 和 TXT 文档。

    返回格式：
    [
        {
            "source": "rag_intro.md",
            "content": "文档正文内容..."
        }
    ]
    """

    documents = []

    # 如果目录不存在，直接报错，方便我们发现问题
    if not raw_docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {raw_docs_dir}")

    # 遍历 raw_docs 目录下的所有文件
    for file_path in raw_docs_dir.iterdir():
        # 只读取 .md 和 .txt 文件
        if file_path.suffix.lower() not in [".md", ".txt"]:
            continue

        # 读取文件内容
        content = file_path.read_text(encoding="utf-8")

        # 去掉前后多余空白
        content = content.strip()

        # 如果文件是空的，就跳过
        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"成功读取 {len(docs)} 个文档：")

    for doc in docs:
        print("-" * 50)
        print(f"文件名：{doc['source']}")
        print(f"字符数：{len(doc['content'])}")
        print(f"内容预览：{doc['content'][:100]}...")