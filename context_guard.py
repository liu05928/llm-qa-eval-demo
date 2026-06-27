import re
from typing import Any, Dict, List


STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "也", "有", "中", "对",
    "什", "么", "为", "何", "如", "哪", "些", "一", "个", "这", "那",
    "吗", "呢", "吧"
}


UNSUPPORTED_PHRASES = [
    "会考哪一道原题",
    "哪一道原题",
    "必考原题",
    "考试原题",
    "考试答案",
    "押题",
    "昨天的考试成绩",
    "学生账号",
    "隐私数据",
    "私人数据",
    "线上服务明天一定会出现的问题",
    "一定会问的问题",
    "2026年最新政策原文",
    "2026 年最新政策原文",
    "最新政策原文",
    "实时股价",
    "实时 gpu 价格",
    "实时GPU价格",
    "真实用户密码",
    "用户密码",
    "生产环境",
    "彩票号码",
    "中奖号码",
]

PRIVATE_TERMS = [
    "隐私",
    "私人",
    "账号",
    "身份证",
    "电话号码",
    "联系方式",
    "密码",
    "考试成绩",
    "成绩",
    "分数",
    "分数线",
]

PRIVATE_SUBJECT_TERMS = [
    "某个学生",
    "学生",
    "同学",
    "个人",
    "用户",
]

FUTURE_CERTAINTY_TERMS = [
    "预测",
    "一定会",
    "必然",
    "肯定",
    "下周",
    "明天",
    "未来",
    "会考",
    "押题",
    "原题",
]

FUTURE_TARGET_TERMS = [
    "考试",
    "原题",
    "线上服务",
    "政策原文",
    "股价",
    "分数线",
    "价格",
    "天气",
    "比赛",
    "彩票",
    "中奖",
]

LATEST_EXTERNAL_TERMS = [
    "政策",
    "政策原文",
    "价格",
    "股价",
    "分数线",
    "融资",
    "招生",
    "录取",
    "天气",
    "比赛",
    "gpu",
]


def _normalize(text: str) -> str:
    return "".join((text or "").lower().split())


def _contains_any(text: str, keywords: List[str]) -> bool:
    lower_text = (text or "").lower()
    normalized_text = _normalize(text)

    return any(
        keyword.lower() in lower_text or _normalize(keyword) in normalized_text
        for keyword in keywords
    )


def bm25_tokenize(text: str) -> List[str]:
    if not text:
        return []

    text = text.lower()

    english_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens = english_tokens + chinese_chars

    return [token for token in tokens if token not in STOPWORDS]


def is_unsupported_fact_request(question: str) -> bool:
    """Return True for requests a static textbook/project KB cannot support."""

    if not question:
        return False

    if _contains_any(question, UNSUPPORTED_PHRASES):
        return True

    if _contains_any(question, PRIVATE_TERMS) and _contains_any(question, PRIVATE_SUBJECT_TERMS):
        return True

    if "推断" in question and _contains_any(question, PRIVATE_TERMS):
        return True

    if _contains_any(question, FUTURE_CERTAINTY_TERMS) and _contains_any(question, FUTURE_TARGET_TERMS):
        return True

    if "最新" in question and _contains_any(question, LATEST_EXTERNAL_TERMS):
        return True

    if _contains_any(question, ["实时", "今天"]) and _contains_any(question, LATEST_EXTERNAL_TERMS):
        return True

    return False


def infer_guard_query_type(question: str) -> str:
    if is_unsupported_fact_request(question):
        return "missing"

    return "general"


def calculate_context_coverage(query: str, chunks: List[Dict[str, Any]]) -> float:
    query_tokens = set(bm25_tokenize(query))

    if not query_tokens:
        return 0.0

    context_tokens = set()

    for chunk in chunks:
        context_tokens.update(bm25_tokenize(chunk.get("content", "")))

    if not context_tokens:
        return 0.0

    hit_count = len(query_tokens.intersection(context_tokens))
    return hit_count / len(query_tokens)


def judge_context(
    original_question: str,
    current_query: str,
    chunks: List[Dict[str, Any]],
    query_type: str,
) -> Dict[str, Any]:
    if query_type == "missing" or is_unsupported_fact_request(original_question):
        return {
            "context_sufficient": False,
            "coverage": 0.0,
            "reason": "问题要求预测未来、获取隐私或查询外部实时事实，当前资料不能直接支持。",
        }

    if not chunks:
        return {
            "context_sufficient": False,
            "coverage": 0.0,
            "reason": "没有召回任何可用片段。",
        }

    original_coverage = calculate_context_coverage(original_question, chunks)
    query_coverage = calculate_context_coverage(current_query, chunks)
    coverage = max(original_coverage, query_coverage)
    source_count = len({chunk.get("source") for chunk in chunks if chunk.get("source")})

    if query_type == "comparison" and source_count < 2 and coverage < 0.35:
        return {
            "context_sufficient": False,
            "coverage": coverage,
            "reason": "对比类问题需要更充分或更多来源的上下文。",
        }

    if coverage < 0.18:
        return {
            "context_sufficient": False,
            "coverage": coverage,
            "reason": "召回片段与问题关键词覆盖不足。",
        }

    return {
        "context_sufficient": True,
        "coverage": coverage,
        "reason": "召回片段覆盖了问题的关键主题，可用于生成回答。",
    }


def build_no_context_answer(question: str, context_reason: str) -> str:
    return (
        "关键词：资料不足、当前知识库、不可编造\n\n"
        "教材依据：当前检索资料没有直接支持该问题的可靠依据，不能把低相关片段当作答案来源。\n\n"
        "回答：资料中未提及足够信息，无法基于当前知识库可靠回答该问题。"
        "我不会补充外部事实或编造结论。\n\n"
        f"学习建议：请补充与“{question}”直接相关的教材、项目说明或权威资料后再提问。"
        f"本次判断原因：{context_reason}\n\n"
        "参考来源：无（未引用低相关检索来源）"
    )
