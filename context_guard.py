import re
from typing import Any, Dict, List, Optional, Set


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
    "未记录实验一定成功",
    "实验一定成功",
    "学校食堂菜单",
    "今天学校食堂菜单",
    "实验室开放时间",
    "家庭住址",
    "作业是不是抄袭",
    "是否请假",
    "是否生病",
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
    "家庭住址",
    "住址",
    "请假",
    "作业",
    "抄袭",
    "生病",
    "健康状况",
]

PRIVATE_SUBJECT_TERMS = [
    "某个学生",
    "某位老师",
    "学生",
    "同学",
    "老师",
    "个人",
    "用户",
]

FUTURE_CERTAINTY_TERMS = [
    "预测",
    "一定会",
    "必然",
    "肯定",
    "一定",
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
    "实验",
    "实验室",
    "开放时间",
    "菜单",
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
    "空气质量",
    "空气质量指数",
    "aqi",
    "指数",
    "菜单",
    "食堂菜单",
    "开放时间",
]

CLAIM_SECTION_HEADERS = [
    "关键词",
    "教材依据",
    "学习建议",
    "参考来源",
]

SCORE_FIELDS = [
    "rerank_combined_score",
    "rerank_score",
    "hybrid_score",
    "bm25_score",
    "dense_score",
]

CLAIM_IGNORE_PHRASES = [
    "根据当前检索资料",
    "本回答只概括",
    "不补充外部事实",
    "复习时先",
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


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number < 0:
        return 0.0

    return min(number, 1.0)


def _risk_result(
    blocked: bool,
    support_level: str = "unknown",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "blocked": blocked,
        "support_level": support_level,
        "reason": reason,
    }


def classify_static_kb_risk(question: str) -> Dict[str, Any]:
    """Classify requests that a static textbook/project KB should not answer."""

    if not question:
        return _risk_result(False)

    if _contains_any(question, PRIVATE_TERMS) and _contains_any(question, PRIVATE_SUBJECT_TERMS):
        return _risk_result(
            True,
            "private",
            "问题涉及个人隐私或账号成绩等信息，当前知识库不能直接支持。",
        )

    if "推断" in question and _contains_any(question, PRIVATE_TERMS):
        return _risk_result(
            True,
            "private",
            "问题要求推断隐私相关信息，当前知识库不能直接支持。",
        )

    if _contains_any(question, FUTURE_CERTAINTY_TERMS) and _contains_any(question, FUTURE_TARGET_TERMS):
        return _risk_result(
            True,
            "future_prediction",
            "问题要求预测未来或给出确定性押题结论，当前资料不能直接支持。",
        )

    if "最新" in question and _contains_any(question, LATEST_EXTERNAL_TERMS):
        return _risk_result(
            True,
            "external_realtime",
            "问题要求查询外部最新事实，静态知识库不能直接支持。",
        )

    if _contains_any(question, ["实时", "今天"]) and _contains_any(question, LATEST_EXTERNAL_TERMS):
        return _risk_result(
            True,
            "external_realtime",
            "问题要求查询实时外部事实，静态知识库不能直接支持。",
        )

    if _contains_any(question, UNSUPPORTED_PHRASES):
        if _contains_any(question, PRIVATE_TERMS + ["密码"]):
            support_level = "private"
        elif _contains_any(question, FUTURE_CERTAINTY_TERMS + ["押题", "原题"]):
            support_level = "future_prediction"
        elif _contains_any(question, LATEST_EXTERNAL_TERMS + ["实时"]):
            support_level = "external_realtime"
        else:
            support_level = "unsupported"

        return _risk_result(
            True,
            support_level,
            "问题命中当前知识库不应回答的高风险事实请求。",
        )

    return _risk_result(False)


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
    return bool(classify_static_kb_risk(question).get("blocked"))


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


def _query_token_set(original_question: str, current_query: str) -> Set[str]:
    tokens = set(bm25_tokenize(original_question))
    tokens.update(bm25_tokenize(current_query))
    return tokens


def _chunk_token_set(chunk: Dict[str, Any]) -> Set[str]:
    return set(bm25_tokenize(chunk.get("content", "")))


def _best_retrieval_score(chunks: List[Dict[str, Any]]) -> float:
    best_score = 0.0

    for chunk in chunks:
        for field in SCORE_FIELDS:
            best_score = max(best_score, _bounded_float(chunk.get(field)))

    return best_score


def calculate_context_evidence(
    original_question: str,
    current_query: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query_tokens = _query_token_set(original_question, current_query)

    if not query_tokens:
        return {
            "coverage": 0.0,
            "best_chunk_coverage": 0.0,
            "evidence_score": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "best_rerank_score": 0.0,
            "source_count": 0,
        }

    context_tokens: Set[str] = set()
    best_chunk_coverage = 0.0

    for chunk in chunks:
        chunk_tokens = _chunk_token_set(chunk)
        context_tokens.update(chunk_tokens)

        if chunk_tokens:
            best_chunk_coverage = max(
                best_chunk_coverage,
                len(query_tokens.intersection(chunk_tokens)) / len(query_tokens),
            )

    matched_terms = sorted(query_tokens.intersection(context_tokens))
    missing_terms = sorted(query_tokens - context_tokens)
    coverage = len(matched_terms) / len(query_tokens)
    source_count = len({chunk.get("source") for chunk in chunks if chunk.get("source")})
    best_rerank_score = _best_retrieval_score(chunks)
    source_bonus = 0.05 if source_count >= 2 else 0.0
    evidence_score = min(
        1.0,
        0.5 * best_chunk_coverage
        + 0.35 * coverage
        + 0.15 * best_rerank_score
        + source_bonus,
    )

    return {
        "coverage": coverage,
        "best_chunk_coverage": best_chunk_coverage,
        "evidence_score": evidence_score,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "best_rerank_score": best_rerank_score,
        "source_count": source_count,
    }


def _build_guard_response(
    context_sufficient: bool,
    coverage: float,
    reason: str,
    support_level: str,
    guard_mode: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = evidence or {}
    evidence_score = round(float(evidence.get("evidence_score", coverage)), 4)

    return {
        "context_sufficient": context_sufficient,
        "coverage": coverage,
        "reason": reason,
        "support_level": support_level,
        "evidence_score": evidence_score,
        "best_chunk_coverage": round(float(evidence.get("best_chunk_coverage", 0.0)), 4),
        "best_rerank_score": round(float(evidence.get("best_rerank_score", 0.0)), 4),
        "matched_terms": evidence.get("matched_terms", []),
        "missing_terms": evidence.get("missing_terms", []),
        "source_count": int(evidence.get("source_count", 0)),
        "guard_mode": guard_mode,
        "guard_details": {
            "support_level": support_level,
            "evidence_score": evidence_score,
            "coverage": round(float(coverage), 4),
            "best_chunk_coverage": round(float(evidence.get("best_chunk_coverage", 0.0)), 4),
            "best_rerank_score": round(float(evidence.get("best_rerank_score", 0.0)), 4),
            "matched_terms": evidence.get("matched_terms", []),
            "missing_terms": evidence.get("missing_terms", []),
            "source_count": int(evidence.get("source_count", 0)),
            "reason": reason,
        },
    }


def _judge_context_v1(
    original_question: str,
    current_query: str,
    chunks: List[Dict[str, Any]],
    query_type: str,
) -> Dict[str, Any]:
    if query_type == "missing" or is_unsupported_fact_request(original_question):
        return _build_guard_response(
            False,
            0.0,
            "问题要求预测未来、获取隐私或查询外部实时事实，当前资料不能直接支持。",
            "unsupported",
            "v1",
        )

    if not chunks:
        return _build_guard_response(
            False,
            0.0,
            "没有召回任何可用片段。",
            "unsupported",
            "v1",
        )

    original_coverage = calculate_context_coverage(original_question, chunks)
    query_coverage = calculate_context_coverage(current_query, chunks)
    coverage = max(original_coverage, query_coverage)
    source_count = len({chunk.get("source") for chunk in chunks if chunk.get("source")})
    evidence = {
        "coverage": coverage,
        "evidence_score": coverage,
        "source_count": source_count,
    }

    if query_type == "comparison" and source_count < 2 and coverage < 0.35:
        return _build_guard_response(
            False,
            coverage,
            "对比类问题需要更充分或更多来源的上下文。",
            "partial",
            "v1",
            evidence,
        )

    if coverage < 0.18:
        return _build_guard_response(
            False,
            coverage,
            "召回片段与问题关键词覆盖不足。",
            "unsupported",
            "v1",
            evidence,
        )

    return _build_guard_response(
        True,
        coverage,
        "召回片段覆盖了问题的关键主题，可用于生成回答。",
        "supported",
        "v1",
        evidence,
    )


def judge_context(
    original_question: str,
    current_query: str,
    chunks: List[Dict[str, Any]],
    query_type: str,
    guard_mode: str = "v2",
) -> Dict[str, Any]:
    if guard_mode == "v1":
        return _judge_context_v1(
            original_question=original_question,
            current_query=current_query,
            chunks=chunks,
            query_type=query_type,
        )

    risk = classify_static_kb_risk(original_question)

    if query_type == "missing" or risk.get("blocked"):
        return _build_guard_response(
            False,
            0.0,
            risk.get("reason")
            or "问题要求预测未来、获取隐私或查询外部实时事实，当前资料不能直接支持。",
            risk.get("support_level") or "unsupported",
            "v2",
        )

    if not chunks:
        return _build_guard_response(
            False,
            0.0,
            "没有召回任何可用片段。",
            "unsupported",
            "v2",
        )

    evidence = calculate_context_evidence(
        original_question=original_question,
        current_query=current_query,
        chunks=chunks,
    )
    coverage = float(evidence.get("coverage", 0.0))
    evidence_score = float(evidence.get("evidence_score", 0.0))
    best_chunk_coverage = float(evidence.get("best_chunk_coverage", 0.0))
    source_count = int(evidence.get("source_count", 0))

    if query_type == "comparison" and source_count < 2 and evidence_score < 0.42:
        return _build_guard_response(
            False,
            coverage,
            "对比类问题需要更充分或更多来源的上下文。",
            "partial",
            "v2",
            evidence,
        )

    if coverage < 0.18 or best_chunk_coverage < 0.16 or evidence_score < 0.24:
        return _build_guard_response(
            False,
            coverage,
            "召回片段与问题关键主题重合不足。",
            "unsupported",
            "v2",
            evidence,
        )

    if coverage < 0.32 or evidence_score < 0.38:
        return _build_guard_response(
            False,
            coverage,
            "召回片段只覆盖了部分问题主题，直接生成存在依据不足风险。",
            "partial",
            "v2",
            evidence,
        )

    return _build_guard_response(
        True,
        coverage,
        "召回片段覆盖了问题关键主题，可用于生成回答。",
        "supported",
        "v2",
        evidence,
    )


def _extract_answer_claims(answer: str) -> List[str]:
    if not answer:
        return []

    answer_text = answer

    if "回答：" in answer_text:
        answer_text = answer_text.split("回答：", 1)[1]

    for header in CLAIM_SECTION_HEADERS:
        marker = f"\n\n{header}："

        if marker in answer_text:
            answer_text = answer_text.split(marker, 1)[0]

    raw_claims = re.split(r"[。！？!?]\s*|\n+", answer_text)
    claims = []

    for claim in raw_claims:
        claim = re.sub(r"^[\-\*\d\.\s、）)]+", "", claim).strip()

        if len(claim) < 8:
            continue

        if _contains_any(claim, ["资料中未提及", "无法基于当前知识库可靠回答"]):
            continue

        if _contains_any(claim, CLAIM_IGNORE_PHRASES):
            continue

        claims.append(claim[:180])

    return claims[:12]


def verify_answer_claims(
    answer: str,
    chunks: List[Dict[str, Any]],
    min_claim_coverage: float = 0.18,
) -> Dict[str, Any]:
    """Lightweight post-generation evidence check for user-visible diagnostics."""

    claims = _extract_answer_claims(answer)

    if not claims:
        return {
            "enabled": True,
            "claim_count": 0,
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
            "claims": [],
            "status": "no_claims",
        }

    verified = []
    supported_count = 0

    for claim in claims:
        coverage = calculate_context_coverage(claim, chunks)
        supported = coverage >= min_claim_coverage

        if supported:
            supported_count += 1

        verified.append(
            {
                "claim": claim,
                "coverage": round(float(coverage), 4),
                "supported": supported,
            }
        )

    unsupported_count = len(verified) - supported_count

    return {
        "enabled": True,
        "claim_count": len(verified),
        "supported_claim_count": supported_count,
        "unsupported_claim_count": unsupported_count,
        "claims": verified,
        "status": "pass" if unsupported_count == 0 else "needs_review",
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
