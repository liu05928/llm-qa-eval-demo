import csv
import os
from pathlib import Path


BASELINE_FILE = Path("eval_results/baseline_eval.csv")
HYBRID_FILE = Path("eval_results/hybrid_rerank_eval.csv")
OUTPUT_FILE = Path("eval_results/failure_cases.md")


def load_csv(file_path: Path):
    """读取 CSV 文件"""
    if not file_path.exists():
        print(f"文件不存在：{file_path}")
        return []

    with file_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_bool(value):
    """把 CSV 中的字符串 True/False/None 转成布尔值"""
    if value is None:
        return None

    value = str(value).strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    if value in ["none", "", "null"]:
        return None

    return None


def get_failed_cases(rows):
    """筛选失败样例"""
    failed_cases = []

    for row in rows:
        source_hit = to_bool(row.get("source_hit"))
        keyword_hit = to_bool(row.get("keyword_hit"))
        has_citation = to_bool(row.get("has_citation"))
        no_context_reject = to_bool(row.get("no_context_reject"))

        question_type = row.get("question_type", "")

        reasons = []

        if source_hit is False:
            reasons.append("检索来源未命中预期文档")

        if keyword_hit is False:
            reasons.append("回答未覆盖预期关键词")

        if has_citation is False:
            reasons.append("回答未返回引用来源")

        if question_type == "missing" and no_context_reject is False:
            reasons.append("资料缺失类问题未正确拒答")

        if reasons:
            item = dict(row)
            item["failure_reasons"] = reasons
            failed_cases.append(item)

    return failed_cases


def classify_failure(case):
    """根据失败原因进行粗分类"""
    reasons = case.get("failure_reasons", [])
    question_type = case.get("question_type", "")

    if any("检索来源未命中" in r for r in reasons):
        return "检索失败"

    if question_type == "missing" and any("未正确拒答" in r for r in reasons):
        return "Prompt 约束不足或幻觉控制不足"

    if any("未覆盖预期关键词" in r for r in reasons):
        return "回答覆盖不足"

    if any("未返回引用来源" in r for r in reasons):
        return "引用来源缺失"

    return "其他问题"


def write_case_section(f, title, rows, max_cases=5):
    """写入某组实验的失败样例"""
    f.write(f"## {title}\n\n")

    failed_cases = get_failed_cases(rows)

    f.write(f"失败样例数量：{len(failed_cases)}\n\n")

    if not failed_cases:
        f.write("未发现明显失败样例。\n\n")
        return

    for i, case in enumerate(failed_cases[:max_cases], start=1):
        failure_type = classify_failure(case)

        f.write(f"### 失败样例 {i}\n\n")
        f.write(f"**问题 ID：** {case.get('id')}\n\n")
        f.write(f"**问题类型：** {case.get('question_type')}\n\n")
        f.write(f"**问题：** {case.get('question')}\n\n")
        f.write(f"**失败类型：** {failure_type}\n\n")
        f.write("**失败原因：**\n\n")

        for reason in case.get("failure_reasons", []):
            f.write(f"- {reason}\n")

        f.write("\n")
        f.write(f"**预期来源：** {case.get('expected_source')}\n\n")
        f.write(f"**实际来源：** {case.get('retrieved_sources')}\n\n")
        f.write(f"**预期关键词：** {case.get('expected_keywords')}\n\n")
        f.write("**模型回答节选：**\n\n")

        answer = case.get("answer", "")
        answer_preview = answer[:300].replace("\n", " ")

        f.write(f"> {answer_preview}\n\n")

        f.write("**可能优化方向：**\n\n")

        if failure_type == "检索失败":
            f.write("- 可尝试改进关键词检索策略，引入 BM25 或 Query Rewrite。\n")
            f.write("- 可扩充知识库内容，增加更多相关资料。\n\n")
        elif failure_type == "Prompt 约束不足或幻觉控制不足":
            f.write("- 可强化 Prompt 中“资料不足时必须拒答”的约束。\n")
            f.write("- 可增加资料缺失类问题的测试样例。\n\n")
        elif failure_type == "回答覆盖不足":
            f.write("- 可优化 Prompt，要求回答覆盖参考资料中的关键概念。\n")
            f.write("- 可调整 Rerank 策略，提高与问题核心词相关的 chunk 排名。\n\n")
        elif failure_type == "引用来源缺失":
            f.write("- 可在 RAG 输出结构中强制返回 sources 字段。\n\n")
        else:
            f.write("- 可结合日志进一步分析失败原因。\n\n")

        f.write("---\n\n")


def main():
    os.makedirs("eval_results", exist_ok=True)

    baseline_rows = load_csv(BASELINE_FILE)
    hybrid_rows = load_csv(HYBRID_FILE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write("# RAG 失败样例分析\n\n")

        f.write("## 1. 分析目的\n\n")
        f.write(
            "本文档基于 RAG 自动评测结果，对基础向量检索方案和 "
            "Dense-Preserving Hybrid Search + Rerank 方案中的失败样例进行归因分析。"
            "分析目标是定位检索失败、回答覆盖不足、引用缺失和资料缺失场景下拒答不稳定等问题，"
            "为后续优化提供依据。\n\n"
        )

        f.write("## 2. 失败类型定义\n\n")
        f.write("| 失败类型 | 含义 |\n")
        f.write("| --- | --- |\n")
        f.write("| 检索失败 | 检索结果未命中预期来源文档 |\n")
        f.write("| 回答覆盖不足 | 回答未覆盖 expected_keywords 中的关键内容 |\n")
        f.write("| 引用来源缺失 | 回答结果未返回 sources 字段 |\n")
        f.write("| Prompt 约束不足或幻觉控制不足 | 资料缺失类问题没有明确拒答 |\n")
        f.write("| 其他问题 | 需要结合日志进一步分析的问题 |\n\n")

        write_case_section(
            f,
            title="3. Baseline：基础向量检索失败样例",
            rows=baseline_rows,
            max_cases=5
        )

        write_case_section(
            f,
            title="4. Dense-Preserving Hybrid Search + Rerank 失败样例",
            rows=hybrid_rows,
            max_cases=5
        )

        f.write("## 5. 总结与后续优化方向\n\n")
        f.write(
            "从失败样例可以看出，RAG 系统的问题不仅来自生成模型，也可能来自检索召回、文本切分、"
            "Prompt 约束和知识库覆盖范围。后续可从以下方向继续优化：\n\n"
        )
        f.write("1. 使用 BM25 替代简单关键词匹配，提高 sparse search 的检索质量；\n")
        f.write("2. 使用 Cross-Encoder Rerank 模型提升候选片段排序效果；\n")
        f.write("3. 增加 Query Rewrite，提高复杂问题和模糊问题的检索效果；\n")
        f.write("4. 扩充教育资料知识库，减少知识库缺失导致的无法回答问题；\n")
        f.write("5. 强化 Prompt 中的拒答约束，进一步降低资料缺失类问题中的幻觉风险。\n")

    print(f"失败样例分析已生成：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()