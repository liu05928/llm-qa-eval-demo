import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


BIG_CHUNKS_FILE = Path("data/chunks/big_chunks.json")


def load_big_chunks(big_chunks_file: Path = BIG_CHUNKS_FILE) -> Dict[str, Dict[str, Any]]:
    """
    读取父级大 chunk，并按 parent_chunk_id 建立索引。
    """

    if not big_chunks_file.exists():
        return {}

    with big_chunks_file.open("r", encoding="utf-8") as f:
        big_chunks = json.load(f)

    return {
        chunk.get("chunk_id"): chunk
        for chunk in big_chunks
        if chunk.get("chunk_id")
    }


def _score_for_order(chunk: Dict[str, Any]) -> float:
    score_fields = [
        "rerank_combined_score",
        "rerank_score",
        "hybrid_score",
        "bm25_score",
        "dense_score",
    ]

    for field in score_fields:
        value = chunk.get(field)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def _copy_retrieval_scores(target: Dict[str, Any], source: Dict[str, Any]):
    for field in [
        "distance",
        "dense_score",
        "bm25_score",
        "hybrid_score",
        "rerank_score",
        "rerank_combined_score",
        "rerank_model",
        "dense_rank",
        "bm25_rank",
    ]:
        if field in source:
            target[field] = source.get(field)


def expand_chunks_to_big_context(
    small_chunks: List[Dict[str, Any]],
    max_big_chunks: int = 3,
    max_total_chars: int = 7000,
    big_chunks_file: Path = BIG_CHUNKS_FILE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Small-to-Big 上下文扩展。

    输入是用于召回的小 chunk；输出是用于回答的大 chunk。多个小 chunk
    命中同一个父级大 chunk 时会去重，并记录触发它的 child_chunk_ids。
    """

    if not small_chunks:
        return [], {
            "enabled": True,
            "expanded": False,
            "reason": "没有可扩展的小 chunk。",
        }

    big_chunk_map = load_big_chunks(big_chunks_file)
    expanded_chunks: List[Dict[str, Any]] = []
    expanded_index: Dict[str, Dict[str, Any]] = {}
    total_chars = 0

    for small_chunk in small_chunks:
        parent_chunk_id = small_chunk.get("parent_chunk_id")
        parent_chunk = big_chunk_map.get(parent_chunk_id)

        if parent_chunk:
            expanded_id = parent_chunk["chunk_id"]
            expanded_content = parent_chunk.get("content", "")
            base_chunk = dict(parent_chunk)
        else:
            expanded_id = small_chunk.get("chunk_id")
            expanded_content = small_chunk.get("content", "")
            base_chunk = dict(small_chunk)
            parent_chunk_id = expanded_id

        if not expanded_id:
            continue

        if expanded_id not in expanded_index:
            if len(expanded_chunks) >= max_big_chunks:
                break

            if total_chars and total_chars + len(expanded_content) > max_total_chars:
                break

            context_chunk = dict(base_chunk)
            context_chunk["chunk_id"] = expanded_id
            context_chunk["source"] = base_chunk.get("source") or small_chunk.get("source")
            context_chunk["content"] = expanded_content
            context_chunk["chunk_type"] = "big"
            context_chunk["parent_chunk_id"] = parent_chunk_id
            context_chunk["context_mode"] = "small_to_big"
            context_chunk["retrieval_type"] = "small_to_big_context"
            context_chunk["trigger_chunk_ids"] = []
            context_chunk["trigger_count"] = 0
            context_chunk["trigger_best_score"] = 0.0
            _copy_retrieval_scores(context_chunk, small_chunk)

            expanded_chunks.append(context_chunk)
            expanded_index[expanded_id] = context_chunk
            total_chars += len(expanded_content)

        context_chunk = expanded_index[expanded_id]
        trigger_chunk_id = small_chunk.get("chunk_id")

        if trigger_chunk_id and trigger_chunk_id not in context_chunk["trigger_chunk_ids"]:
            context_chunk["trigger_chunk_ids"].append(trigger_chunk_id)

        context_chunk["trigger_count"] = len(context_chunk["trigger_chunk_ids"])

        small_score = _score_for_order(small_chunk)

        if small_score >= context_chunk.get("trigger_best_score", 0.0):
            context_chunk["trigger_best_score"] = small_score
            context_chunk["trigger_chunk_id"] = trigger_chunk_id
            _copy_retrieval_scores(context_chunk, small_chunk)

    summary = {
        "enabled": True,
        "expanded": True,
        "small_chunk_count": len(small_chunks),
        "big_chunk_count": len(expanded_chunks),
        "max_big_chunks": max_big_chunks,
        "max_total_chars": max_total_chars,
        "total_context_chars": total_chars,
    }

    return expanded_chunks, summary
