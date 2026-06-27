#!/usr/bin/env python3
"""Run fixed generation-quality evaluation for OpenAI-compatible chat APIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "你是一名可信教育问答助手。回答时必须严格依据用户提供的参考资料，"
    "不要引入资料外信息。若资料不足以回答问题，应明确说明资料不足，"
    "不要编造。除非题目只要求检索改写，否则尽量使用以下结构：关键词、"
    "教材依据、回答、学习建议、参考来源。参考来源必须与资料中的来源一致。"
)

REFUSAL_CUES = (
    "资料不足",
    "资料中未提及",
    "参考资料未提及",
    "无法根据",
    "不能根据",
    "不能从资料",
    "不编造",
    "没有足够信息",
    "无法判断",
)

SOURCE_PATTERN = re.compile(r"(?:science_textbooks|docs|data|training)/[^\s，。；;、)）]+?\.md")
SECTION_LABELS = ("关键词：", "教材依据：", "资料依据：", "回答：", "学习建议：", "参考来源：")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        default="data/sft_v2/quality_eval_questions.json",
        help="Path to the fixed eval questions JSON.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001/v1/chat/completions",
        help="OpenAI-compatible chat completions URL, or the /v1 base URL.",
    )
    parser.add_argument("--model", default="qwen2.5-3b-edu", help="Model name sent to the API.")
    parser.add_argument("--name", required=True, help="Short run name, e.g. v2_qlora.")
    parser.add_argument("--output-dir", default="eval_results/sft_generation", help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional question limit for smoke tests.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def normalize_chat_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def load_questions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return {"count": len(payload)}, payload

    if not isinstance(payload, dict) or "questions" not in payload:
        raise ValueError(f"Unsupported questions format: {path}")

    questions = payload["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list in {path}")

    meta = {k: v for k, v in payload.items() if k != "questions"}
    return meta, questions


def call_chat_api(
    *,
    chat_url: str,
    model: str,
    user_input: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {json.dumps(result, ensure_ascii=False)[:1000]}") from exc


def contains_any(text: str, cues: tuple[str, ...] | list[str]) -> bool:
    return any(cue in text for cue in cues)


def score_answer(item: dict[str, Any], answer: str) -> dict[str, Any]:
    expected_keywords = item.get("expected_keywords") or []
    expected_reference = item.get("expected_reference") or item.get("source") or ""

    hit_keywords = [kw for kw in expected_keywords if kw and kw in answer]
    missed_keywords = [kw for kw in expected_keywords if kw and kw not in answer]
    keyword_total = len(expected_keywords)
    keyword_score = len(hit_keywords) / keyword_total if keyword_total else 0.0

    extracted_sources = sorted(set(SOURCE_PATTERN.findall(answer)))
    unexpected_sources = [src for src in extracted_sources if src != expected_reference]

    reference_present = bool(expected_reference and expected_reference in answer)
    reference_mismatch = bool(unexpected_sources)
    refusal_detected = contains_any(answer, REFUSAL_CUES)
    is_refusal_case = item.get("type") == "refusal"

    structure_complete = (
        "关键词" in answer
        and ("教材依据" in answer or "资料依据" in answer)
        and "学习建议" in answer
        and ("参考来源" in answer or "来源" in answer)
    )
    repeated_section = any(answer.count(label) > 1 for label in SECTION_LABELS)

    unsupported_refusal = (not is_refusal_case) and refusal_detected and keyword_score < 0.5
    possible_fabrication = reference_mismatch or (is_refusal_case and not refusal_detected)

    return {
        "keyword_score": round(keyword_score, 4),
        "keyword_any_hit": bool(hit_keywords),
        "keyword_all_hit": keyword_total > 0 and len(hit_keywords) == keyword_total,
        "hit_keywords": hit_keywords,
        "missed_keywords": missed_keywords,
        "citation_complete": reference_present,
        "reference_mismatch": reference_mismatch,
        "extracted_sources": extracted_sources,
        "unexpected_sources": unexpected_sources,
        "structure_complete": structure_complete,
        "repeated_section": repeated_section,
        "refusal_detected": refusal_detected,
        "refusal_correct": refusal_detected if is_refusal_case else None,
        "unsupported_refusal": unsupported_refusal,
        "possible_fabrication": possible_fabrication,
    }


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def rate(values: list[bool]) -> float:
    return round(sum(1 for item in values if item) / len(values), 4) if values else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [item for item in results if not item.get("error")]
    non_refusal = [item for item in ok_results if item["type"] != "refusal"]
    refusal = [item for item in ok_results if item["type"] == "refusal"]

    by_type: dict[str, dict[str, Any]] = {}
    for item in ok_results:
        by_type.setdefault(item["type"], {"items": []})["items"].append(item)

    for item_type, group in by_type.items():
        rows = group.pop("items")
        group.update(
            {
                "count": len(rows),
                "avg_keyword_score": average([row["keyword_score"] for row in rows]),
                "keyword_any_hit_rate": rate([row["keyword_any_hit"] for row in rows]),
                "keyword_all_hit_rate": rate([row["keyword_all_hit"] for row in rows]),
                "citation_complete_rate": rate([row["citation_complete"] for row in rows]),
                "structure_complete_rate": rate([row["structure_complete"] for row in rows]),
                "repeated_section_rate": rate([row["repeated_section"] for row in rows]),
                "reference_mismatch_rate": rate([row["reference_mismatch"] for row in rows]),
                "possible_fabrication_rate": rate([row["possible_fabrication"] for row in rows]),
            }
        )
        if item_type == "refusal":
            group["refusal_correct_rate"] = rate([row["refusal_correct"] for row in rows])

    return {
        "question_count": len(results),
        "completed_count": len(ok_results),
        "error_count": len(results) - len(ok_results),
        "avg_keyword_score": average([row["keyword_score"] for row in ok_results]),
        "keyword_any_hit_rate": rate([row["keyword_any_hit"] for row in ok_results]),
        "keyword_all_hit_rate": rate([row["keyword_all_hit"] for row in ok_results]),
        "citation_complete_rate": rate([row["citation_complete"] for row in ok_results]),
        "structure_complete_rate": rate([row["structure_complete"] for row in ok_results]),
        "repeated_section_rate": rate([row["repeated_section"] for row in ok_results]),
        "refusal_correct_rate": rate([row["refusal_correct"] for row in refusal]),
        "unsupported_refusal_rate": rate([row["unsupported_refusal"] for row in non_refusal]),
        "reference_mismatch_rate": rate([row["reference_mismatch"] for row in ok_results]),
        "possible_fabrication_rate": rate([row["possible_fabrication"] for row in ok_results]),
        "avg_latency_s": average([row["latency_s"] for row in ok_results]),
        "by_type": by_type,
    }


def write_outputs(output_dir: Path, run_name: str, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_name}_summary.json"
    details_path = output_dir / f"{run_name}_details.jsonl"
    csv_path = output_dir / f"{run_name}_details.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with details_path.open("w", encoding="utf-8") as f:
        for item in payload["details"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    fields = [
        "id",
        "type",
        "domain",
        "question",
        "keyword_score",
        "keyword_any_hit",
        "keyword_all_hit",
        "citation_complete",
        "structure_complete",
        "repeated_section",
        "refusal_detected",
        "refusal_correct",
        "unsupported_refusal",
        "reference_mismatch",
        "possible_fabrication",
        "latency_s",
        "expected_reference",
        "hit_keywords",
        "missed_keywords",
        "answer",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in payload["details"]:
            row = dict(item)
            row["hit_keywords"] = " | ".join(row.get("hit_keywords") or [])
            row["missed_keywords"] = " | ".join(row.get("missed_keywords") or [])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    questions_meta, questions = load_questions(Path(args.questions))
    if args.limit > 0:
        questions = questions[: args.limit]

    chat_url = normalize_chat_url(args.base_url)
    output_dir = Path(args.output_dir)
    started_at = datetime.now().isoformat(timespec="seconds")

    details: list[dict[str, Any]] = []
    total = len(questions)

    print(f"Run: {args.name}")
    print(f"Questions: {total}")
    print(f"Endpoint: {chat_url}")

    for index, item in enumerate(questions, start=1):
        item_id = item.get("id", f"q{index:03d}")
        print(f"[{index:03d}/{total:03d}] {item_id} {item.get('type', '')}", flush=True)
        start = time.monotonic()
        error = ""
        answer = ""
        try:
            answer = call_chat_api(
                chat_url=chat_url,
                model=args.model,
                user_input=item.get("input") or item.get("question", ""),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            score = score_answer(item, answer)
        except Exception as exc:  # noqa: BLE001 - preserve eval progress on one failed sample.
            error = str(exc)
            score = {
                "keyword_score": 0.0,
                "keyword_any_hit": False,
                "keyword_all_hit": False,
                "hit_keywords": [],
                "missed_keywords": item.get("expected_keywords") or [],
                "citation_complete": False,
                "reference_mismatch": False,
                "extracted_sources": [],
                "unexpected_sources": [],
                "structure_complete": False,
                "repeated_section": False,
                "refusal_detected": False,
                "refusal_correct": None,
                "unsupported_refusal": False,
                "possible_fabrication": False,
            }

        latency_s = round(time.monotonic() - start, 3)
        detail = {
            "id": item_id,
            "type": item.get("type", ""),
            "domain": item.get("domain", ""),
            "question": item.get("question", ""),
            "expected_reference": item.get("expected_reference") or item.get("source", ""),
            "expected_keywords": item.get("expected_keywords") or [],
            "answer": answer,
            "latency_s": latency_s,
            "error": error,
            **score,
        }
        details.append(detail)

    summary = summarize(details)
    payload = {
        "run_name": args.name,
        "model": args.model,
        "base_url": chat_url,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "questions_meta": questions_meta,
        "settings": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        },
        "summary": summary,
        "details": details,
    }
    write_outputs(output_dir, args.name, payload)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output_dir / f'{args.name}_summary.json'}")


if __name__ == "__main__":
    main()
