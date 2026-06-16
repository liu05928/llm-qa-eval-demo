import csv
import json
from pathlib import Path
from typing import Dict, List


TEST_FILE = "data/science_rag_test_questions.json"
SMALL_EVAL_FILE = "eval_results/science_small_context_eval.csv"
SMALL_TO_BIG_EVAL_FILE = "eval_results/science_small_to_big_eval.csv"
OUTPUT_FILE = "eval_results/science_failure_cases.md"


def _to_bool(value):
    if value in (True, False, None):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    return None


def _read_rows(file_path: str, mode: str) -> List[Dict[str, str]]:
    path = Path(file_path)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                **row,
                "analysis_mode": mode,
            }
            for row in reader
        ]


def _classify_row(row: Dict[str, str]) -> List[str]:
    reasons = []
    question_type = row.get("question_type", "")
    expected_no_context_reject = _to_bool(row.get("expected_no_context_reject"))

    source_hit = _to_bool(row.get("source_hit"))
    exact_keyword_hit = _to_bool(row.get("exact_keyword_hit"))
    semantic_keyword_hit = _to_bool(row.get("semantic_keyword_hit"))
    has_citation = _to_bool(row.get("has_citation"))
    no_context_reject = _to_bool(row.get("no_context_reject"))

    if expected_no_context_reject or question_type == "missing":
        if no_context_reject is not True:
            reasons.append("资料缺失题未明确拒答，存在把无关资料拼成答案的风险。")
        return reasons

    if source_hit is False:
        reasons.append("检索未命中预期教材来源。")

    if semantic_keyword_hit is False:
        reasons.append("回答没有覆盖预期核心关键词，可能是回答覆盖不足。")
    elif exact_keyword_hit is False and semantic_keyword_hit is True:
        reasons.append("严格关键词未命中但同义关键词命中，属于表达变体造成的假阴性。")

    if has_citation is False:
        reasons.append("回答没有返回引用来源。")

    return reasons


def _write_case(f, row: Dict[str, str], reasons: List[str]):
    f.write(
        f"### {row.get('analysis_mode')} / 第 {row.get('id')} 题\n\n"
    )
    f.write(f"**问题：** {row.get('question')}\n\n")
    f.write(f"**问题类型：** {row.get('question_type')}\n\n")
    f.write(f"**预期来源：** {row.get('expected_source') or '无，资料缺失题'}\n\n")
    f.write(f"**召回来源：** {row.get('retrieved_sources') or '无'}\n\n")
    f.write("**原因：**\n\n")

    for reason in reasons:
        f.write(f"- {reason}\n")

    f.write("\n")

    answer = row.get("answer", "")
    if answer:
        preview = answer.replace("\n", " ")[:260]
        f.write(f"**回答摘要：** {preview}\n\n")


def _load_expected_question_count(test_file: str = TEST_FILE) -> int:
    path = Path(test_file)

    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as f:
        return len(json.load(f))


def write_science_failure_report(
    small_file: str = SMALL_EVAL_FILE,
    small_to_big_file: str = SMALL_TO_BIG_EVAL_FILE,
    output_file: str = OUTPUT_FILE,
) -> Dict[str, int]:
    rows = []
    rows.extend(_read_rows(small_file, "small"))
    rows.extend(_read_rows(small_to_big_file, "small_to_big"))
    expected_question_count = _load_expected_question_count()
    expected_row_count = expected_question_count * 2 if expected_question_count else 0
    is_stale = bool(expected_row_count and len(rows) != expected_row_count)

    hard_failures = []
    soft_observations = []

    for row in rows:
        reasons = _classify_row(row)

        if not reasons:
            continue

        item = {
            "row": row,
            "reasons": reasons,
        }

        if any("假阴性" in reason for reason in reasons) and len(reasons) == 1:
            soft_observations.append(item)
        else:
            hard_failures.append(item)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# 科学教材 RAG 失败样例与边界分析\n\n")
        f.write("## 汇总\n\n")
        f.write(f"- 评测记录数：{len(rows)}\n")
        if expected_row_count:
            f.write(f"- 当前题集预期记录数：{expected_row_count}\n")
        f.write(f"- 硬性失败数：{len(hard_failures)}\n")
        f.write(f"- 软性观察数：{len(soft_observations)}\n\n")

        if is_stale:
            f.write("> 注意：当前 CSV 记录数与最新题集规模不一致，请重新运行 `science_long_text_eval_runner.py` 后再使用本报告作为最终实验结论。\n\n")

        if hard_failures:
            f.write("## 硬性失败样例\n\n")
            for item in hard_failures:
                _write_case(f, item["row"], item["reasons"])
        else:
            f.write("## 硬性失败样例\n\n")
            f.write("本次评测未发现硬性失败。\n\n")

        if soft_observations:
            f.write("## 软性观察\n\n")
            for item in soft_observations:
                _write_case(f, item["row"], item["reasons"])

        f.write("## 系统边界\n\n")
        f.write("- 当前科学教材知识库来自抽样导入，不代表覆盖所有初中科学知识点。\n")
        f.write("- Small-to-Big 增加回答上下文，但不会改变第一步小 chunk 召回结果。\n")
        f.write("- 资料缺失拒答依赖 Prompt 约束和 Agent 上下文判断，后续可用更多缺失题持续压测。\n")
        f.write("- 严格关键词命中适合排查措辞变化，同义关键词命中更适合判断实际语义覆盖。\n")

    return {
        "rows": len(rows),
        "expected_rows": expected_row_count,
        "stale": is_stale,
        "hard_failures": len(hard_failures),
        "soft_observations": len(soft_observations),
    }


def main():
    summary = write_science_failure_report()
    print(summary)


if __name__ == "__main__":
    main()
