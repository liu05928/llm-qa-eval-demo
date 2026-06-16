import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SOURCE_DIR = Path("../ch-3/knowledge_base_builder/data/textbooks/初中科学/沪教版初中科学")
DEFAULT_TARGET_DIR = Path("data/raw_docs/science_textbooks")
DEFAULT_MANIFEST_FILE = Path("data/science_textbook_manifest.json")
GENERATED_FILE_PATTERN = re.compile(r"^\d{3}_.+\.md$")

METADATA_FIELDS = [
    "textbook_id",
    "grade",
    "semester",
    "publisher",
    "school_level",
    "chapter",
    "section",
    "content_type",
    "token_count",
]

FIELD_LABELS = {
    "textbook_id": "教材",
    "grade": "年级",
    "semester": "册次",
    "publisher": "出版社",
    "school_level": "学段",
    "chapter": "章节",
    "section": "小节",
    "content_type": "内容类型",
    "token_count": "原始 token 数",
}


def parse_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text.strip()

    parts = text.split("---", 2)

    if len(parts) < 3:
        return {}, text.strip()

    metadata_text = parts[1]
    body = parts[2].strip()
    metadata: Dict[str, str] = {}

    for raw_line in metadata_text.splitlines():
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        metadata[key.strip()] = value

    return metadata, body


def sanitize_filename(value: str, max_length: int = 90) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", value).strip("_")

    if not cleaned:
        return "science_doc"

    return cleaned[:max_length].rstrip("_")


def load_candidate_docs(source_dir: Path) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    for file_path in sorted(source_dir.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(text)

        if len(body) < 120:
            continue

        candidates.append(
            {
                "path": file_path,
                "metadata": metadata,
                "body": body,
                "textbook_id": metadata.get("textbook_id", "unknown"),
            }
        )

    return candidates


def sample_evenly(items: List[Dict[str, object]], count: int) -> List[Dict[str, object]]:
    if count <= 0:
        return []

    if count >= len(items):
        return items

    if count == 1:
        return [items[0]]

    last_index = len(items) - 1
    indices = [
        round(index * last_index / (count - 1))
        for index in range(count)
    ]
    deduped_indices = sorted(set(indices))

    while len(deduped_indices) < count:
        for index in range(len(items)):
            if index not in deduped_indices:
                deduped_indices.append(index)
                break

    return [items[index] for index in sorted(deduped_indices[:count])]


def select_balanced_docs(candidates: List[Dict[str, object]], max_docs: int) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for item in candidates:
        grouped[str(item["textbook_id"])].append(item)

    for items in grouped.values():
        items.sort(key=lambda item: str(item["path"]))

    textbook_ids = sorted(grouped)
    base_quota = max_docs // len(textbook_ids) if textbook_ids else 0
    remainder = max_docs % len(textbook_ids) if textbook_ids else 0
    sampled_groups: Dict[str, List[Dict[str, object]]] = {}

    for index, textbook_id in enumerate(textbook_ids):
        quota = base_quota + (1 if index < remainder else 0)
        sampled_groups[textbook_id] = sample_evenly(grouped[textbook_id], quota)

    selected: List[Dict[str, object]] = []
    while len(selected) < max_docs:
        added = False

        for textbook_id in textbook_ids:
            if not sampled_groups[textbook_id]:
                continue

            selected.append(sampled_groups[textbook_id].pop(0))
            added = True

            if len(selected) >= max_docs:
                break

        if not added:
            break

    return selected


def clear_generated_files(target_dir: Path) -> int:
    deleted_count = 0

    if not target_dir.exists():
        return deleted_count

    for file_path in target_dir.glob("*.md"):
        if GENERATED_FILE_PATTERN.match(file_path.name):
            file_path.unlink()
            deleted_count += 1

    return deleted_count


def build_imported_text(metadata: Dict[str, str], body: str, source_path: Path) -> str:
    textbook_id = metadata.get("textbook_id", source_path.stem)
    chapter = metadata.get("chapter", "")
    title = f"初中科学教材：{textbook_id}"

    if chapter:
        title = f"{title} - {chapter}"

    lines = [
        f"# {title}",
        "",
        "## 教材元信息",
    ]

    for field in METADATA_FIELDS:
        value = metadata.get(field)

        if value:
            lines.append(f"- {FIELD_LABELS[field]}：{value}")

    lines.extend(
        [
            f"- 原始文件：{source_path.as_posix()}",
            "",
            "## 教材正文",
            body.strip(),
            "",
        ]
    )

    return "\n".join(lines)


def import_science_textbooks(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    target_dir: Path = DEFAULT_TARGET_DIR,
    manifest_file: Path = DEFAULT_MANIFEST_FILE,
    max_docs: int = 80,
    clear_target: bool = True,
) -> Dict[str, object]:
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在：{source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    deleted_count = clear_generated_files(target_dir) if clear_target else 0

    candidates = load_candidate_docs(source_dir)
    selected_docs = select_balanced_docs(candidates, max_docs=max_docs)
    written_files: List[str] = []
    chapter_counter: Counter = Counter()
    textbook_counter: Counter = Counter()

    for index, item in enumerate(selected_docs, start=1):
        source_path = item["path"]
        metadata = item["metadata"]
        body = item["body"]
        target_name = f"{index:03d}_{sanitize_filename(source_path.stem)}.md"
        target_path = target_dir / target_name
        imported_text = build_imported_text(metadata, body, source_path)

        target_path.write_text(imported_text, encoding="utf-8")
        written_files.append(target_path.as_posix())
        chapter_counter.update([metadata.get("chapter", "unknown")])
        textbook_counter.update([metadata.get("textbook_id", "unknown")])

    manifest = {
        "domain": "初中科学教材",
        "source_dir": source_dir.as_posix(),
        "target_dir": target_dir.as_posix(),
        "candidate_count": len(candidates),
        "imported_count": len(written_files),
        "deleted_generated_files": deleted_count,
        "max_docs": max_docs,
        "textbook_distribution": dict(textbook_counter.most_common()),
        "top_chapters": dict(chapter_counter.most_common(20)),
        "written_files": written_files,
    }

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def main():
    parser = argparse.ArgumentParser(description="导入 ch-3 初中科学教材语料到当前 RAG 知识库。")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    manifest = import_science_textbooks(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        manifest_file=args.manifest_file,
        max_docs=args.max_docs,
        clear_target=not args.keep_existing,
    )

    print(f"候选文档数：{manifest['candidate_count']}")
    print(f"清理旧导入文件数：{manifest['deleted_generated_files']}")
    print(f"导入文档数：{manifest['imported_count']}")
    print(f"目标目录：{manifest['target_dir']}")
    print(f"清单文件：{args.manifest_file.as_posix()}")


if __name__ == "__main__":
    main()
