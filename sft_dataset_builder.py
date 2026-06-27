import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCIENCE_TEST_FILE = Path("data/science_rag_test_questions.json")
RAG_TEST_FILE = Path("data/rag_test_questions.json")
MEMORY_TEST_FILE = Path("data/memory_eval_questions.json")
CHUNKS_FILE = Path("data/chunks/chunks.json")
BIG_CHUNKS_FILE = Path("data/chunks/big_chunks.json")
DEFAULT_OUTPUT_DIR = Path("data/sft")

GROUNDING_INSTRUCTION = (
    "你是一名初中科学智能助教。请严格根据参考资料回答学生问题；"
    "如果参考资料不足，请明确说明“资料中未提及”，不要编造。"
)

CITATION_INSTRUCTION = (
    "请根据给定教材资料回答问题，并在答案末尾写出参考来源。"
    "不要使用参考资料之外的事实。"
)

REWRITE_INSTRUCTION = (
    "请把学生的口语化追问改写成适合知识库检索的一句话。"
    "只输出改写后的问题，不要回答。"
)

REFUSAL_INSTRUCTION = (
    "你是一名可信教育问答助手。请判断参考资料是否足以回答问题；"
    "资料不足时必须拒答并说明原因。"
)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def stable_bucket(key: str, buckets: int = 10) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def choose_split(source_key: str) -> str:
    bucket = stable_bucket(source_key)

    if bucket < 8:
        return "train"

    if bucket == 8:
        return "dev"

    return "test"


def split_sentences(text: str) -> List[str]:
    candidates = re.split(r"[。！？!?]\s*|\n+", text or "")
    sentences = []

    for sentence in candidates:
        sentence = normalize_space(sentence)

        if 12 <= len(sentence) <= 180:
            sentences.append(sentence)

    return sentences


def keyword_hit(sentence: str, keywords: Iterable[str]) -> bool:
    normalized = sentence.lower()
    return any(keyword and keyword.lower() in normalized for keyword in keywords)


def extract_evidence_sentences(
    context: str,
    keywords: List[str],
    max_sentences: int = 3,
) -> List[str]:
    sentences = split_sentences(context)
    selected = [
        sentence
        for sentence in sentences
        if keyword_hit(sentence, keywords)
    ]

    if len(selected) < max_sentences:
        for sentence in sentences:
            if sentence not in selected:
                selected.append(sentence)

            if len(selected) >= max_sentences:
                break

    return selected[:max_sentences]


def build_context_indexes() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    big_chunks = load_json(BIG_CHUNKS_FILE)
    small_chunks = load_json(CHUNKS_FILE)

    big_by_source: Dict[str, Dict[str, Any]] = {}
    small_by_source: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in big_chunks:
        source = chunk.get("source")
        if source and source not in big_by_source:
            big_by_source[source] = chunk

    for chunk in small_chunks:
        source = chunk.get("source")

        if source:
            small_by_source.setdefault(source, []).append(chunk)

    return big_by_source, small_by_source


def get_context_for_source(
    source: Optional[str],
    big_by_source: Dict[str, Dict[str, Any]],
    small_by_source: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, str]:
    if not source:
        return "", ""

    if source in big_by_source:
        chunk = big_by_source[source]
        return chunk.get("content", ""), chunk.get("chunk_id", "")

    chunks = small_by_source.get(source, [])
    context = "\n\n".join(chunk.get("content", "") for chunk in chunks[:3])
    chunk_ids = ",".join(chunk.get("chunk_id", "") for chunk in chunks[:3])
    return context, chunk_ids


def make_input(question: str, context: str, source: Optional[str]) -> str:
    if context:
        return (
            f"学生问题：{question}\n\n"
            f"参考资料：\n[来源：{source}]\n{truncate_text(context, 1600)}"
        )

    return (
        f"学生问题：{question}\n\n"
        "参考资料：未检索到可以回答该问题的教材片段。"
    )


def build_grounded_output(
    question: str,
    context: str,
    source: str,
    keywords: List[str],
    citation: bool = True,
) -> str:
    evidence = extract_evidence_sentences(context, keywords)
    keyword_text = "、".join(keywords[:4]) if keywords else "教材资料"

    lines = [
        "根据参考资料，可以这样回答：",
        f"1. 这个问题的核心关键词是：{keyword_text}。",
    ]

    if evidence:
        for idx, sentence in enumerate(evidence, start=2):
            lines.append(f"{idx}. 教材依据：{sentence}。")
    else:
        lines.append("2. 资料中给出了与问题相关的教材片段，但没有形成完整定义，需要结合片段谨慎解释。")

    lines.append("学习建议：复习时先抓住教材中的关键词，再把它们和具体例子或实验现象联系起来。")

    if citation:
        lines.append(f"参考来源：{source}")

    return "\n".join(lines)


def build_refusal_output(question: str) -> str:
    return (
        "资料中未提及足以回答该问题的信息，因此不能根据当前教材资料给出确定答案。\n"
        "我不会补充外部事实或编造结论。建议补充对应教材章节、官方资料或更新后的知识库后再查询。"
    )


def make_record(
    sample_id: str,
    sample_type: str,
    instruction: str,
    input_text: str,
    output_text: str,
    source: Optional[str],
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "type": sample_type,
        "source": source or "",
        "sample": {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        },
    }


def build_qa_records(
    items: List[Dict[str, Any]],
    prefix: str,
    big_by_source: Dict[str, Dict[str, Any]],
    small_by_source: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    records = []

    for item in items:
        qid = item.get("id")
        question = item.get("question", "").strip()
        source = item.get("expected_source")
        keywords = item.get("expected_keywords", [])
        question_type = item.get("question_type", "general")
        context, chunk_ids = get_context_for_source(source, big_by_source, small_by_source)

        if not question:
            continue

        if source and context:
            grounded_input = make_input(question, context, source)
            output = build_grounded_output(
                question=question,
                context=context,
                source=source,
                keywords=keywords,
                citation=True,
            )
            records.append(
                make_record(
                    sample_id=f"{prefix}_{qid}_grounded",
                    sample_type=f"grounded_qa:{question_type}",
                    instruction=GROUNDING_INSTRUCTION,
                    input_text=grounded_input,
                    output_text=output,
                    source=source,
                )
            )

            citation_input = (
                f"{grounded_input}\n\n"
                f"可用来源标识：source={source}; chunk_id={chunk_ids or 'unknown'}"
            )
            records.append(
                make_record(
                    sample_id=f"{prefix}_{qid}_citation",
                    sample_type=f"citation_qa:{question_type}",
                    instruction=CITATION_INSTRUCTION,
                    input_text=citation_input,
                    output_text=output,
                    source=source,
                )
            )
        else:
            records.append(
                make_record(
                    sample_id=f"{prefix}_{qid}_refusal",
                    sample_type="refusal",
                    instruction=REFUSAL_INSTRUCTION,
                    input_text=make_input(question, "", None),
                    output_text=build_refusal_output(question),
                    source=None,
                )
            )

    return records


def build_memory_rewrite_records(memory_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for case in memory_cases:
        title = case.get("title", "")
        case_id = case.get("case_id", "")

        for turn in case.get("turns", []):
            if not turn.get("expected_memory_used"):
                continue

            question = turn.get("question", "").strip()
            keywords = turn.get("expected_resolved_keywords", [])

            if not question or not keywords:
                continue

            rewritten = f"{' '.join(keywords)}：{question}"
            input_text = (
                f"多轮对话主题：{turn.get('expected_topic', title)}\n"
                f"学生追问：{question}\n"
                f"需要保留的主题词：{'、'.join(keywords)}"
            )
            records.append(
                make_record(
                    sample_id=f"{case_id}_{turn.get('turn_id')}_rewrite",
                    sample_type="query_rewrite",
                    instruction=REWRITE_INSTRUCTION,
                    input_text=input_text,
                    output_text=rewritten,
                    source=turn.get("expected_source"),
                )
            )

    return records


def build_extra_refusal_records() -> List[Dict[str, Any]]:
    questions = [
        "2026年最新中考科学分数线是多少？",
        "某个学生昨天的考试成绩是多少？",
        "火星明天的天气预报是多少？",
        "资料里有没有给出某个模型训练使用的全部 GPU 数量？",
        "这份教材有没有说明 DeepSeek 最新模型的完整参数量？",
        "请根据当前资料预测下周股票走势。",
    ]
    records = []

    for idx, question in enumerate(questions, start=1):
        records.append(
            make_record(
                sample_id=f"extra_refusal_{idx:03d}",
                sample_type="refusal",
                instruction=REFUSAL_INSTRUCTION,
                input_text=make_input(question, "", None),
                output_text=build_refusal_output(question),
                source=None,
            )
        )

    return records


def split_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"train": [], "dev": [], "test": []}

    for record in records:
        source_key = record.get("source") or record["id"]
        split = choose_split(source_key)
        grouped[split].append(record)

    for split_records_ in grouped.values():
        random.Random(42).shuffle(split_records_)

    return grouped


def write_jsonl(path: Path, records: List[Dict[str, Any]], sample_only: bool):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record["sample"] if sample_only else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize(records: List[Dict[str, Any]], splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    for record in records:
        sample_type = record.get("type", "unknown")
        source = record.get("source") or "no_source"
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "total": len(records),
        "split_counts": {
            split: len(items)
            for split, items in splits.items()
        },
        "type_counts": dict(sorted(type_counts.items())),
        "source_count": len(source_counts),
        "top_sources": sorted(
            source_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
        "format": "alpaca_jsonl",
        "fields": ["instruction", "input", "output"],
        "note": "Generated locally from existing RAG/science evaluation sets and chunk evidence. Review before cloud training.",
    }


def validate_samples(records: List[Dict[str, Any]]):
    seen_ids = set()

    for record in records:
        record_id = record.get("id")

        if not record_id:
            raise ValueError("存在缺少 id 的样本。")

        if record_id in seen_ids:
            raise ValueError(f"样本 id 重复: {record_id}")

        seen_ids.add(record_id)
        sample = record.get("sample", {})

        for field in ["instruction", "input", "output"]:
            value = sample.get(field, "")

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"样本 {record_id} 缺少字段: {field}")


def build_dataset(output_dir: Path) -> Dict[str, Any]:
    science_items = load_json(SCIENCE_TEST_FILE)
    rag_items = load_json(RAG_TEST_FILE)
    memory_cases = load_json(MEMORY_TEST_FILE)
    big_by_source, small_by_source = build_context_indexes()

    records = []
    records.extend(
        build_qa_records(
            science_items,
            prefix="science",
            big_by_source=big_by_source,
            small_by_source=small_by_source,
        )
    )
    records.extend(
        build_qa_records(
            rag_items,
            prefix="rag",
            big_by_source=big_by_source,
            small_by_source=small_by_source,
        )
    )
    records.extend(build_memory_rewrite_records(memory_cases))
    records.extend(build_extra_refusal_records())
    validate_samples(records)

    splits = split_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_items in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_items, sample_only=True)
        write_jsonl(output_dir / f"{split}_with_metadata.jsonl", split_items, sample_only=False)

    write_jsonl(output_dir / "all_with_metadata.jsonl", records, sample_only=False)
    manifest = summarize(records, splits)
    dump_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="从现有教材/RAG评测题和 chunks 构造 LLaMA Factory SFT JSONL 数据。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认 data/sft",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_dataset(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
