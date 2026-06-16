import csv
import json
import os
from typing import Any, Dict, List

from experiment_runner import (
    check_has_citation,
    check_no_context_reject,
    check_source_hit,
    normalize_sources,
)
from rag_pipeline import rag_answer
from science_failure_analyzer import write_science_failure_report


TEST_FILE = "data/science_rag_test_questions.json"
EVAL_DIR = "eval_results"
SMALL_EVAL_FILE = "eval_results/science_small_context_eval.csv"
SMALL_TO_BIG_EVAL_FILE = "eval_results/science_small_to_big_eval.csv"
SUMMARY_FILE = "eval_results/science_small_to_big_summary.json"
REPORT_FILE = "eval_results/science_small_to_big_report.md"
FAILURE_REPORT_FILE = "eval_results/science_failure_cases.md"

SUMMARY_METRICS = [
    "source_hit_rate",
    "exact_keyword_hit_rate",
    "semantic_keyword_hit_rate",
    "has_citation_rate",
    "no_context_reject_rate",
    "avg_answer_length",
    "avg_context_chars",
    "avg_small_chunk_count",
    "avg_big_chunk_count",
    "avg_trigger_count",
]

KEYWORD_ALIASES = {
    "固态": ["固态", "固体形态", "固体状态", "固体"],
    "液态": ["液态", "液体形态", "液体状态", "液体"],
    "气态": ["气态", "气体形态", "气体状态", "气体", "水蒸气"],
    "牛顿第一定律": ["牛顿第一定律", "惯性定律"],
    "二氧化碳": ["二氧化碳", "CO2", "CO₂"],
    "氧气": ["氧气", "O2", "O₂"],
    "受力面积": ["受力面积", "接触面积"],
    "生物多样性": ["生物多样性", "物种多样性", "生态多样性"],
    "昼夜": ["昼夜", "昼夜交替", "白天和黑夜"],
    "帕斯卡": ["帕斯卡", "Pa"],
    "石蕊": ["石蕊", "紫色石蕊"],
    "酚酞": ["酚酞", "无色酚酞"],
    "变红": ["变红", "红色"],
    "变蓝": ["变蓝", "蓝色"],
    "氢氧化钙": ["氢氧化钙", "熟石灰", "消石灰"],
    "电磁铁": ["电磁铁", "螺线管"],
    "线圈": ["线圈", "螺线管", "线圈疏密"],
    "铁心": ["铁心", "铁芯"],
    "大气压": ["大气压", "气压"],
    "沸点": ["沸点", "水的沸点"],
    "高压锅": ["高压锅", "锅内气压"],
    "杠杆": ["杠杆", "撬棒", "天平"],
    "动力臂": ["动力臂", "l1", "l_1"],
    "阻力臂": ["阻力臂", "l2", "l_2"],
    "生产者": ["生产者", "绿色植物"],
    "消费者": ["消费者", "初级消费者", "次级消费者"],
    "分解者": ["分解者", "细菌", "真菌"],
    "蒸腾作用": ["蒸腾作用", "水分的散失"],
    "叶片": ["叶片", "叶子"],
    "稀有气体": ["稀有气体", "惰性气体"],
    "有色的光": ["有色的光", "紫蓝色光", "粉红色光", "红光"],
    "卵细胞": ["卵细胞", "卵"],
    "精子": ["精子", "sperm"],
    "电压": ["电压", "安全电压"],
    "电流": ["电流", "安全电流"],
}


def load_questions(file_path: str = TEST_FILE) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: List[Dict[str, Any]], output_file: str):
    if not rows:
        return

    os.makedirs(EVAL_DIR, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calc_rate(values: List[Any]) -> float:
    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return sum(1 for value in valid_values if value is True) / len(valid_values)


def calc_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)

    if total == 0:
        return {
            "total": 0,
            "source_hit_rate": 0,
            "exact_keyword_hit_rate": 0,
            "semantic_keyword_hit_rate": 0,
            "has_citation_rate": 0,
            "no_context_reject_rate": 0,
            "avg_answer_length": 0,
            "avg_context_chars": 0,
            "avg_small_chunk_count": 0,
            "avg_big_chunk_count": 0,
            "avg_trigger_count": 0,
        }

    return {
        "total": total,
        "source_hit_rate": calc_rate([row["source_hit"] for row in rows]),
        "exact_keyword_hit_rate": calc_rate([row["exact_keyword_hit"] for row in rows]),
        "semantic_keyword_hit_rate": calc_rate([row["semantic_keyword_hit"] for row in rows]),
        "has_citation_rate": calc_rate([row["has_citation"] for row in rows]),
        "no_context_reject_rate": calc_rate([row["no_context_reject"] for row in rows]),
        "avg_answer_length": sum(row["answer_length"] for row in rows) / total,
        "avg_context_chars": sum(row["context_chars"] for row in rows) / total,
        "avg_small_chunk_count": sum(row["small_chunk_count"] for row in rows) / total,
        "avg_big_chunk_count": sum(row["big_chunk_count"] for row in rows) / total,
        "avg_trigger_count": sum(row["trigger_count"] for row in rows) / total,
    }


def format_metric(metric: str, value: float) -> str:
    if metric.endswith("_rate"):
        return f"{value:.2%}"

    return f"{value:.1f}"


def calc_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        metric: after.get(metric, 0) - before.get(metric, 0)
        for metric in SUMMARY_METRICS
    }


def _aliases_for_keyword(keyword: str) -> List[str]:
    aliases = KEYWORD_ALIASES.get(keyword, [keyword])
    return list(dict.fromkeys([keyword] + aliases))


def _hit_keyword(keyword: str, answer: str, use_aliases: bool) -> bool:
    candidates = _aliases_for_keyword(keyword) if use_aliases else [keyword]
    return any(candidate in answer for candidate in candidates)


def check_keyword_hit_with_aliases(
    expected_keywords: List[str],
    answer: str,
    use_aliases: bool,
) -> bool:
    """
    命中一半及以上关键词，记为 True。

    use_aliases=False: 严格字面命中。
    use_aliases=True: 支持预设同义词/别名，降低表达变体造成的假阴性。
    """

    if not expected_keywords:
        return None

    hit_count = 0

    for keyword in expected_keywords:
        if _hit_keyword(keyword, answer, use_aliases=use_aliases):
            hit_count += 1

    return hit_count / len(expected_keywords) >= 0.5


def collect_keyword_hits(
    expected_keywords: List[str],
    answer: str,
    use_aliases: bool,
) -> List[str]:
    hits = []

    for keyword in expected_keywords:
        if _hit_keyword(keyword, answer, use_aliases=use_aliases):
            hits.append(keyword)

    return hits


def run_context_eval(context_mode: str, output_file: str) -> Dict[str, Any]:
    questions = load_questions()
    rows: List[Dict[str, Any]] = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])
        question_type = item.get("question_type", "")
        expected_no_context_reject = bool(
            item.get("expected_no_context_reject")
            or question_type == "missing"
        )

        print("=" * 80)
        print(f"正在评测：{context_mode} | 第 {qid} 题：{question}")

        try:
            result = rag_answer(
                question=question,
                top_k=3,
                retriever_mode="bm25_hybrid",
                candidate_k=10,
                use_rerank=True,
                context_mode=context_mode,
            )

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            retrieved_chunks = result.get("retrieved_chunks", [])
            small_chunks = result.get("small_retrieved_chunks", [])
            long_context = result.get("long_context", {})
            retrieved_sources = normalize_sources(sources)
            has_citation = (
                None
                if expected_no_context_reject
                else check_has_citation(sources)
            )
            no_context_reject = (
                check_no_context_reject("missing", answer)
                if expected_no_context_reject
                else None
            )
            context_chars = sum(
                len(chunk.get("content", ""))
                for chunk in retrieved_chunks
            )
            trigger_count = sum(
                int(chunk.get("trigger_count") or 0)
                for chunk in retrieved_chunks
            )

            row = {
                "id": qid,
                "question": question,
                "question_type": question_type,
                "context_mode": context_mode,
                "retriever_mode": result.get("retriever_mode"),
                "expected_source": expected_source,
                "expected_no_context_reject": expected_no_context_reject,
                "retrieved_sources": "|".join(retrieved_sources),
                "source_hit": check_source_hit(expected_source, retrieved_sources),
                "expected_keywords": "|".join(expected_keywords),
                "exact_keyword_hit": check_keyword_hit_with_aliases(
                    expected_keywords,
                    answer,
                    use_aliases=False,
                ),
                "semantic_keyword_hit": check_keyword_hit_with_aliases(
                    expected_keywords,
                    answer,
                    use_aliases=True,
                ),
                "exact_keyword_hits": "|".join(
                    collect_keyword_hits(
                        expected_keywords,
                        answer,
                        use_aliases=False,
                    )
                ),
                "semantic_keyword_hits": "|".join(
                    collect_keyword_hits(
                        expected_keywords,
                        answer,
                        use_aliases=True,
                    )
                ),
                "has_citation": has_citation,
                "no_context_reject": no_context_reject,
                "retrieved_chunk_count": len(retrieved_chunks),
                "small_chunk_count": (
                    len(small_chunks)
                    if context_mode == "small_to_big"
                    else len(retrieved_chunks)
                ),
                "big_chunk_count": int(long_context.get("big_chunk_count") or 0),
                "trigger_count": trigger_count,
                "context_chars": context_chars,
                "avg_context_chars_per_chunk": (
                    context_chars / len(retrieved_chunks)
                    if retrieved_chunks
                    else 0
                ),
                "answer_length": len(answer),
                "long_context": json.dumps(long_context, ensure_ascii=False),
                "answer": answer,
            }
            rows.append(row)

        except Exception as exc:
            print(f"评测失败：{exc}")
            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "context_mode": context_mode,
                "retriever_mode": "bm25_hybrid",
                "expected_source": expected_source,
                "expected_no_context_reject": expected_no_context_reject,
                "retrieved_sources": "",
                "source_hit": check_source_hit(expected_source, []),
                "expected_keywords": "|".join(expected_keywords),
                "exact_keyword_hit": False if expected_keywords else None,
                "semantic_keyword_hit": False if expected_keywords else None,
                "exact_keyword_hits": "",
                "semantic_keyword_hits": "",
                "has_citation": False if not expected_no_context_reject else None,
                "no_context_reject": False if expected_no_context_reject else None,
                "retrieved_chunk_count": 0,
                "small_chunk_count": 0,
                "big_chunk_count": 0,
                "trigger_count": 0,
                "context_chars": 0,
                "avg_context_chars_per_chunk": 0,
                "answer_length": 0,
                "long_context": "{}",
                "answer": f"ERROR: {exc}",
            })

    save_csv(rows, output_file)
    return calc_summary(rows)


def write_report(
    small_summary: Dict[str, Any],
    small_to_big_summary: Dict[str, Any],
    delta: Dict[str, Any],
):
    metric_names = {
        "source_hit_rate": "来源命中率",
        "exact_keyword_hit_rate": "严格关键词命中率",
        "semantic_keyword_hit_rate": "同义关键词命中率",
        "has_citation_rate": "引用完整率",
        "no_context_reject_rate": "资料缺失拒答率",
        "avg_answer_length": "平均回答长度",
        "avg_context_chars": "平均上下文字符数",
        "avg_small_chunk_count": "平均小 chunk 数",
        "avg_big_chunk_count": "平均父级大 chunk 数",
        "avg_trigger_count": "平均触发小 chunk 数",
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# 科学教材 Small-to-Big 对比实验报告\n\n")
        f.write("## 实验设置\n\n")
        f.write(f"- 测试集：`data/science_rag_test_questions.json`，共 {small_summary.get('total', 0)} 条初中科学教材问题。\n")
        f.write("- 召回策略：`retriever_mode=bm25_hybrid`，`top_k=3`，`candidate_k=10`，`use_rerank=True`。\n")
        f.write("- 对照组：`context_mode=small`，小 chunk 直接用于回答。\n")
        f.write("- 实验组：`context_mode=small_to_big`，小 chunk 召回后扩展到父级大 chunk 用于回答。\n\n")
        f.write("关键词评测同时保留两种口径：严格关键词命中只做字面匹配；同义关键词命中支持少量教材领域别名，例如 `固态≈固体形态`、`气态≈气体形态`。\n")
        f.write("资料缺失题不参与来源命中率、关键词命中率和引用完整率统计，单独统计资料缺失拒答率。\n\n")

        f.write("## 指标对比\n\n")
        f.write("| 指标 | small | small_to_big | 差值 |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        for metric in SUMMARY_METRICS:
            f.write(
                "| "
                f"{metric_names[metric]} | "
                f"{format_metric(metric, small_summary.get(metric, 0))} | "
                f"{format_metric(metric, small_to_big_summary.get(metric, 0))} | "
                f"{format_metric(metric, delta.get(metric, 0))} |\n"
            )

        f.write("\n## 结论\n\n")
        f.write("- Small-to-Big 不改变小 chunk 的召回入口，因此来源命中率主要反映检索质量。\n")
        f.write("- Small-to-Big 显著增加回答阶段可用上下文长度，适合教材、政策、公文制度等长文本资料。\n")
        f.write("- 同义关键词命中用于降低表达变体造成的假阴性，严格关键词命中仍保留用于排查回答措辞变化。\n")
        f.write("- 资料缺失拒答率用于验证系统边界意识，避免把无关教材片段强行拼成答案。\n")
        f.write("- 页面和日志会保留 `trigger_chunk_ids`，可追溯父级大段落由哪些小 chunk 命中触发。\n\n")

        f.write("## 结果文件\n\n")
        f.write(f"- `{SMALL_EVAL_FILE}`\n")
        f.write(f"- `{SMALL_TO_BIG_EVAL_FILE}`\n")
        f.write(f"- `{SUMMARY_FILE}`\n")
        f.write(f"- `{FAILURE_REPORT_FILE}`\n")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    small_summary = run_context_eval(
        context_mode="small",
        output_file=SMALL_EVAL_FILE,
    )
    small_to_big_summary = run_context_eval(
        context_mode="small_to_big",
        output_file=SMALL_TO_BIG_EVAL_FILE,
    )
    delta = calc_delta(small_summary, small_to_big_summary)

    summary = {
        "small": small_summary,
        "small_to_big": small_to_big_summary,
        "delta": delta,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(
        small_summary=small_summary,
        small_to_big_summary=small_to_big_summary,
        delta=delta,
    )
    write_science_failure_report(
        small_file=SMALL_EVAL_FILE,
        small_to_big_file=SMALL_TO_BIG_EVAL_FILE,
        output_file=FAILURE_REPORT_FILE,
    )

    print("\nSmall-to-Big 对比评测完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV：{SMALL_EVAL_FILE}")
    print(f"CSV：{SMALL_TO_BIG_EVAL_FILE}")
    print(f"Summary：{SUMMARY_FILE}")
    print(f"Report：{REPORT_FILE}")
    print(f"Failure Report：{FAILURE_REPORT_FILE}")


if __name__ == "__main__":
    main()
