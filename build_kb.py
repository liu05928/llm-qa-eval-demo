import argparse
import json
from pathlib import Path
from typing import Any, Dict

from config import USE_MOCK
from document_loader import load_documents
from domain_kb_importer import (
    DEFAULT_MANIFEST_FILE,
    DEFAULT_SOURCE_DIR,
    DEFAULT_TARGET_DIR,
    import_science_textbooks,
)
from rag_pipeline import rag_answer
from science_long_text_eval_runner import main as run_science_long_text_eval
from text_splitter import (
    BIG_CHUNKS_FILE,
    CHUNKS_FILE,
    DEFAULT_BIG_CHUNK_OVERLAP,
    DEFAULT_BIG_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_OVERLAP,
    DEFAULT_SMALL_CHUNK_SIZE,
    save_big_chunks,
    save_chunks,
    split_documents_small_to_big,
)
from vector_store import VectorStore


SMOKE_QUESTION = "压力和压强有什么关系？"


def print_step(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def count_science_docs(documents) -> int:
    return sum(
        1
        for doc in documents
        if str(doc.get("source", "")).startswith("science_textbooks/")
    )


def build_chunks() -> Dict[str, Any]:
    documents = load_documents()
    small_chunks, big_chunks = split_documents_small_to_big(
        documents=documents,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
        small_chunk_overlap=DEFAULT_SMALL_CHUNK_OVERLAP,
        big_chunk_size=DEFAULT_BIG_CHUNK_SIZE,
        big_chunk_overlap=DEFAULT_BIG_CHUNK_OVERLAP,
    )

    save_chunks(small_chunks)
    save_big_chunks(big_chunks)

    return {
        "document_count": len(documents),
        "science_document_count": count_science_docs(documents),
        "small_chunk_count": len(small_chunks),
        "big_chunk_count": len(big_chunks),
        "chunks_file": CHUNKS_FILE.as_posix(),
        "big_chunks_file": BIG_CHUNKS_FILE.as_posix(),
    }


def build_vector_index() -> Dict[str, Any]:
    vector_store = VectorStore()
    chunks = vector_store.load_chunks()
    vector_store.build_index()

    return {
        "indexed_chunk_count": len(chunks),
        "use_mock": USE_MOCK,
    }


def run_smoke_test() -> Dict[str, Any]:
    result = rag_answer(
        question=SMOKE_QUESTION,
        top_k=3,
        retriever_mode="bm25_hybrid",
        candidate_k=10,
        use_rerank=True,
        context_mode="small_to_big",
    )

    return {
        "question": SMOKE_QUESTION,
        "context_mode": result.get("context_mode"),
        "retriever_mode": result.get("retriever_mode"),
        "source_count": len(result.get("sources", [])),
        "first_source": (
            result.get("sources", [{}])[0].get("source")
            if result.get("sources")
            else None
        ),
        "long_context": result.get("long_context", {}),
    }


def write_summary(summary: Dict[str, Any], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="一键构建教育领域垂直知识库：导入语料、切分 small/big chunks、构建向量索引并可选评测。"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("eval_results/build_kb_summary.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary: Dict[str, Any] = {
        "use_mock": USE_MOCK,
        "import": None,
        "chunks": None,
        "index": None,
        "smoke": None,
        "eval": None,
    }

    print_step("运行配置")
    print(f"USE_MOCK={USE_MOCK}")
    print(f"source_dir={args.source_dir}")
    print(f"target_dir={args.target_dir}")
    print(f"max_docs={args.max_docs}")

    if not args.skip_import:
        print_step("1. 导入垂直领域教材语料")
        import_summary = import_science_textbooks(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            manifest_file=args.manifest_file,
            max_docs=args.max_docs,
            clear_target=not args.keep_existing,
        )
        summary["import"] = import_summary
        print(f"候选文档数：{import_summary['candidate_count']}")
        print(f"导入文档数：{import_summary['imported_count']}")
    else:
        print_step("1. 跳过语料导入")

    print_step("2. 生成 small/big chunks")
    chunk_summary = build_chunks()
    summary["chunks"] = chunk_summary
    print(f"文档数：{chunk_summary['document_count']}")
    print(f"科学教材文档数：{chunk_summary['science_document_count']}")
    print(f"small chunk 数：{chunk_summary['small_chunk_count']}")
    print(f"big chunk 数：{chunk_summary['big_chunk_count']}")

    if not args.skip_index:
        print_step("3. 构建 Chroma 向量索引")
        index_summary = build_vector_index()
        summary["index"] = index_summary
        print(f"写入索引 small chunk 数：{index_summary['indexed_chunk_count']}")
    else:
        print_step("3. 跳过向量索引构建")

    if not args.skip_smoke:
        print_step("4. Small-to-Big 冒烟测试")
        smoke_summary = run_smoke_test()
        summary["smoke"] = smoke_summary
        print(f"问题：{smoke_summary['question']}")
        print(f"检索模式：{smoke_summary['retriever_mode']}")
        print(f"上下文模式：{smoke_summary['context_mode']}")
        print(f"首个来源：{smoke_summary['first_source']}")
        print(f"Long context：{smoke_summary['long_context']}")
    else:
        print_step("4. 跳过冒烟测试")

    if args.run_eval:
        print_step("5. 运行科学教材 small vs small_to_big 对比评测")
        run_science_long_text_eval()
        summary["eval"] = {
            "summary_file": "eval_results/science_small_to_big_summary.json",
            "report_file": "eval_results/science_small_to_big_report.md",
        }
    else:
        print_step("5. 跳过评测")

    write_summary(summary, args.summary_file)
    print_step("构建完成")
    print(f"构建摘要：{args.summary_file}")


if __name__ == "__main__":
    main()
