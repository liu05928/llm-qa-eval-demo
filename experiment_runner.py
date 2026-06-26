import json
import csv
import os
import re
import time
from typing import List, Dict, Any

from rag_pipeline import rag_answer


TEST_FILE = "data/rag_test_questions.json"
EVAL_DIR = "eval_results"

NO_KEYWORD_FILE = "eval_results/no_keyword_dense_rerank_eval.csv"
BM25_FILE = "eval_results/bm25_hybrid_rerank_eval.csv"
COMPARISON_SUMMARY_FILE = "eval_results/bm25_comparison_summary.json"
COMPARISON_REPORT_FILE = "eval_results/bm25_comparison_report.md"

SUMMARY_METRICS = [
    "source_hit_rate",
    "keyword_hit_rate",
    "has_citation_rate",
    "no_context_reject_rate",
    "avg_answer_length",
    "avg_retrieved_chunk_count",
]

KEYWORD_ALIASES = {
    "提示词": ["提示词", "prompt", "提示"],
    "模型输出": ["模型输出", "模型生成", "生成结果", "输出结果", "模型回答", "回答内容"],
    "约束": ["约束", "限制", "规范", "控制", "要求"],
    "教学设计": ["教学设计", "教学目标", "教学活动", "备课", "教案"],
    "答疑": ["答疑", "解答疑问", "回答问题", "智能问答", "问题解答"],
    "学习资源": ["学习资源", "学习材料", "学习资料", "课程资源", "资源推荐"],
    "无法确定": ["无法确定", "不能确定", "无法得知", "无法找到", "无法提供", "无法获取", "无法从资料中确定", "资料不足"],
    "资料中未提及": ["资料中未提及", "资料未提及", "未提及", "没有提到", "没有提供", "未提供", "没有相关信息", "未找到相关信息", "不包含"],
    "当前资料": ["当前资料", "现有资料", "参考资料", "给定资料", "提供的参考资料", "根据现有资料", "根据提供的资料"],
}


def load_test_questions(file_path: str = TEST_FILE) -> List[Dict[str, Any]]:
    """读取测试集"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sources(sources):
    """把 sources 统一成 source 文件名列表"""
    if not sources:
        return []

    result = []

    for item in sources:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            source = item.get("source")
            if source:
                result.append(source)

    return result


def check_source_hit(expected_source, retrieved_sources):
    """
    判断来源是否命中。
    资料缺失类问题 expected_source 为 None，不参与来源命中率统计。
    """
    if expected_source is None:
        return None

    return expected_source in retrieved_sources


def normalize_keyword_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def keyword_tokens(text: str) -> List[str]:
    text = (text or "").lower()
    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_tokens + chinese_chars


def aliases_for_keyword(keyword: str) -> List[str]:
    aliases = KEYWORD_ALIASES.get(keyword, [])
    return list(dict.fromkeys([keyword] + aliases))


def is_keyword_covered(keyword: str, answer: str) -> bool:
    normalized_answer = normalize_keyword_text(answer)

    for candidate in aliases_for_keyword(keyword):
        if normalize_keyword_text(candidate) in normalized_answer:
            return True

    tokens = keyword_tokens(keyword)

    if len(tokens) < 3:
        return False

    answer_tokens = set(keyword_tokens(answer))

    return all(token in answer_tokens for token in tokens)


def collect_keyword_hits(expected_keywords: List[str], answer: str) -> List[str]:
    return [
        keyword
        for keyword in expected_keywords
        if is_keyword_covered(keyword, answer)
    ]


def check_keyword_hit(expected_keywords, answer):
    """
    判断关键词是否命中，支持少量同义表达和短语 token 覆盖。
    命中一半及以上关键词，记为 True。
    """
    if not expected_keywords:
        return None

    hit_count = len(collect_keyword_hits(expected_keywords, answer))

    return hit_count / len(expected_keywords) >= 0.5


def check_has_citation(sources):
    """判断是否返回引用来源"""
    return len(sources) > 0


def check_no_context_reject(question_type, answer):
    """
    判断资料缺失类问题是否拒答。
    只对 question_type == missing 的问题统计。
    """
    if question_type != "missing":
        return None

    refusal_phrases = [
        "无法确定",
        "不能确定",
        "无法得知",
        "无法找到",
        "资料中未提及",
        "当前资料",
        "没有提到",
        "没有相关信息",
        "未找到相关信息",
        "无法从资料",
        "不能从资料",
        "未提供相关信息",
        "无法提供",
        "无法获取",
        "不包含",
        "不可能包含",
        "超出资料",
        "不在资料",
        "不知道",
        "无法回答"
    ]

    return any(phrase in answer for phrase in refusal_phrases)


def run_single_eval(
    output_file: str,
    retriever_mode: str,
    top_k: int = 3,
    candidate_k: int = 10,
    use_rerank: bool = True,
    max_retries: int = 2,
    retry_sleep_seconds: int = 3,
):
    """
    运行单组实验。
    """
    os.makedirs(EVAL_DIR, exist_ok=True)

    questions = load_test_questions()
    rows = []

    for item in questions:
        qid = item.get("id")
        question = item.get("question", "")
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_keywords", [])
        question_type = item.get("question_type", "")

        print("=" * 80)
        print(f"正在评测：{retriever_mode} | 第 {qid} 题：{question}")

        try:
            for attempt in range(max_retries + 1):
                try:
                    result = rag_answer(
                        question=question,
                        top_k=top_k,
                        retriever_mode=retriever_mode,
                        candidate_k=candidate_k,
                        use_rerank=use_rerank,
                    )
                    break
                except Exception as retry_error:
                    if attempt >= max_retries:
                        raise

                    print(
                        f"第 {qid} 题调用失败，"
                        f"{retry_sleep_seconds} 秒后重试 "
                        f"({attempt + 1}/{max_retries})：{retry_error}"
                    )
                    time.sleep(retry_sleep_seconds)

            answer = result.get("answer", "")
            sources = result.get("sources", [])
            retrieved_chunks = result.get("retrieved_chunks", [])

            retrieved_sources = normalize_sources(sources)

            source_hit = check_source_hit(expected_source, retrieved_sources)
            keyword_hits = collect_keyword_hits(expected_keywords, answer)
            keyword_hit = check_keyword_hit(expected_keywords, answer)
            has_citation = check_has_citation(sources)
            no_context_reject = check_no_context_reject(question_type, answer)
            if question_type == "missing" and no_context_reject:
                keyword_hit = True
                keyword_hits.append("no_context_reject")

            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "retriever_mode": retriever_mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "use_rerank": use_rerank,
                "expected_source": expected_source,
                "retrieved_sources": "|".join(retrieved_sources),
                "source_hit": source_hit,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hits": "|".join(keyword_hits),
                "keyword_hit": keyword_hit,
                "has_citation": has_citation,
                "no_context_reject": no_context_reject,
                "retrieved_chunk_count": len(retrieved_chunks),
                "answer_length": len(answer),
                "answer": answer
            })

        except Exception as e:
            print(f"评测失败：{e}")

            rows.append({
                "id": qid,
                "question": question,
                "question_type": question_type,
                "retriever_mode": retriever_mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "use_rerank": use_rerank,
                "expected_source": expected_source,
                "retrieved_sources": "",
                "source_hit": False if expected_source is not None else None,
                "expected_keywords": "|".join(expected_keywords),
                "keyword_hits": "",
                "keyword_hit": False,
                "has_citation": False,
                "no_context_reject": False if question_type == "missing" else None,
                "retrieved_chunk_count": 0,
                "answer_length": 0,
                "answer": f"ERROR: {e}"
            })

    save_csv(rows, output_file)
    summary = calc_summary(rows)

    print("\n评测完成：", output_file)
    print_summary(summary)

    return summary


def save_csv(rows, output_file):
    """保存 CSV"""
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calc_rate(values):
    """计算 True 比例，忽略 None"""
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        return 0

    true_count = sum(1 for v in valid_values if v is True)
    return true_count / len(valid_values)


def calc_summary(rows):
    """计算汇总指标"""
    total = len(rows)

    return {
        "total": total,
        "source_hit_rate": calc_rate([r["source_hit"] for r in rows]),
        "keyword_hit_rate": calc_rate([r["keyword_hit"] for r in rows]),
        "has_citation_rate": calc_rate([r["has_citation"] for r in rows]),
        "no_context_reject_rate": calc_rate([r["no_context_reject"] for r in rows]),
        "avg_answer_length": sum(r["answer_length"] for r in rows) / total if total else 0,
        "avg_retrieved_chunk_count": sum(r["retrieved_chunk_count"] for r in rows) / total if total else 0
    }


def calc_delta(before_summary, after_summary):
    """计算 BM25 相比 no-keyword 方案的指标变化。"""
    return {
        metric: after_summary.get(metric, 0) - before_summary.get(metric, 0)
        for metric in SUMMARY_METRICS
    }


def print_summary(summary):
    print(f"测试问题数：{summary['total']}")
    print(f"来源命中率：{summary['source_hit_rate']:.2%}")
    print(f"关键词命中率：{summary['keyword_hit_rate']:.2%}")
    print(f"引用完整率：{summary['has_citation_rate']:.2%}")
    print(f"无资料拒答率：{summary['no_context_reject_rate']:.2%}")
    print(f"平均回答长度：{summary['avg_answer_length']:.1f}")
    print(f"平均检索片段数：{summary['avg_retrieved_chunk_count']:.1f}")


def format_metric(metric, value):
    """格式化报告中的指标。"""
    if metric.endswith("_rate"):
        return f"{value:.2%}"

    return f"{value:.1f}"


def write_comparison_report(no_keyword_summary, bm25_summary, delta):
    """生成 no-keyword 与 BM25 的 Markdown 对比报告。"""
    metric_names = {
        "source_hit_rate": "来源命中率",
        "keyword_hit_rate": "关键词命中率",
        "has_citation_rate": "引用完整率",
        "no_context_reject_rate": "无资料拒答率",
        "avg_answer_length": "平均回答长度",
        "avg_retrieved_chunk_count": "平均检索片段数",
    }

    with open(COMPARISON_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# BM25 对比实验报告\n\n")
        f.write("## 实验设置\n\n")
        f.write("- No Keyword：`retriever_mode=dense_rerank`，向量召回后直接 Rerank。\n")
        f.write("- BM25 Hybrid：`retriever_mode=bm25_hybrid`，向量召回 + BM25 召回 + RRF 融合后 Rerank。\n")
        f.write("- 两组均使用 `top_k=3`、`candidate_k=10`、`use_rerank=True`。\n\n")

        f.write("## 指标对比\n\n")
        f.write("| 指标 | No Keyword | BM25 Hybrid | BM25 - No Keyword |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        for metric in SUMMARY_METRICS:
            f.write(
                "| "
                f"{metric_names[metric]} | "
                f"{format_metric(metric, no_keyword_summary.get(metric, 0))} | "
                f"{format_metric(metric, bm25_summary.get(metric, 0))} | "
                f"{format_metric(metric, delta.get(metric, 0))} |\n"
            )

        f.write("\n## 结果文件\n\n")
        f.write(f"- `{NO_KEYWORD_FILE}`\n")
        f.write(f"- `{BM25_FILE}`\n")
        f.write(f"- `{COMPARISON_SUMMARY_FILE}`\n")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("\n开始运行 No Keyword：Dense Rerank")
    no_keyword_summary = run_single_eval(
        output_file=NO_KEYWORD_FILE,
        retriever_mode="dense_rerank",
        top_k=3,
        candidate_k=10,
        use_rerank=True
    )

    print("\n开始运行 BM25 Hybrid + Rerank")
    bm25_summary = run_single_eval(
        output_file=BM25_FILE,
        retriever_mode="bm25_hybrid",
        top_k=3,
        candidate_k=10,
        use_rerank=True
    )

    delta = calc_delta(no_keyword_summary, bm25_summary)

    experiment_summary = {
        "dense_rerank_no_keyword": no_keyword_summary,
        "bm25_hybrid_rerank": bm25_summary,
        "delta_bm25_minus_no_keyword": delta,
    }

    with open(COMPARISON_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)

    write_comparison_report(
        no_keyword_summary=no_keyword_summary,
        bm25_summary=bm25_summary,
        delta=delta,
    )

    print("\n" + "=" * 80)
    print("对比实验完成")
    print(f"No Keyword 结果：{NO_KEYWORD_FILE}")
    print(f"BM25 Hybrid 结果：{BM25_FILE}")
    print(f"汇总结果：{COMPARISON_SUMMARY_FILE}")
    print(f"对比报告：{COMPARISON_REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
