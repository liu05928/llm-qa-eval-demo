import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

from agent_memory import ensure_memory
from agent_state import AgentState, AgentTraceStep
from context_guard import (
    build_no_context_answer,
    is_unsupported_fact_request,
    judge_context,
    verify_answer_claims,
)
from config import USE_MOCK
from llm_client import call_llm
from prompt_templates import build_rag_prompt
from rag_pipeline import (
    build_chunk_log_list,
    build_sources,
    format_context,
    retrieve_chunks,
)


AGENT_LOG_FILE = Path("logs/agent_trace_log.json")

QUERY_TYPE_LABELS = {
    "concept": "概念解释",
    "project_concept": "项目知识库概念",
    "comparison": "对比分析",
    "learning_advice": "学习建议",
    "missing": "资料缺失候选",
    "vague": "模糊问题",
    "general": "普通问题",
}

DOMAIN_TERMS = [
    "rag",
    "agent",
    "prompt",
    "prompt engineering",
    "prompt a/b",
    "prompt a/b 测试",
    "embedding",
    "rerank",
    "bm25",
    "hybrid",
    "rrf",
    "向量",
    "向量数据库",
    "向量检索",
    "知识库",
    "检索",
    "重排序",
    "大模型",
    "教育",
    "幻觉",
    "微调",
    "sft",
    "dpo",
    "lora",
    "初中科学",
    "科学教材",
    "教材",
    "实验室",
    "观察",
    "显微镜",
    "细胞",
    "植物",
    "动物",
    "质量",
    "密度",
    "速度",
    "运动",
    "光",
    "声音",
    "透镜",
    "眼睛",
    "视觉",
    "听觉",
    "地球",
    "自转",
    "公转",
    "星空",
    "水",
    "溶解",
    "溶液",
    "呼吸",
    "土壤",
    "果实",
    "干果",
    "肉果",
    "磁化",
    "磁体",
    "磁场",
    "能量",
    "燃烧",
    "电流",
    "电路",
    "力",
    "浮力",
    "压强",
]

PROJECT_DOMAIN_TERMS = [
    "rag",
    "agent",
    "workflow",
    "langgraph",
    "prompt",
    "prompt engineering",
    "embedding",
    "rerank",
    "bm25",
    "hybrid",
    "rrf",
    "small-to-big",
    "small to big",
    "context guard",
    "向量数据库",
    "向量检索",
    "知识库",
    "检索增强",
    "智能体",
    "多智能体",
    "a/b 测试",
    "ab测试",
    "重排序",
    "大模型",
    "微调",
    "sft",
    "dpo",
    "lora",
    "qlora",
]

PROJECT_QUERY_CUES = [
    "什么是",
    "是什么",
    "解释",
    "定义",
    "原理",
    "机制",
    "作用",
    "意思",
    "是什么意思",
    "为什么",
    "有哪些",
    "关系",
    "特点",
    "说明了什么",
    "基本流程",
    "基本循环",
    "区别",
    "对比",
    "相比",
    "不同",
    "差异",
    "怎么学",
    "如何学习",
    "学习建议",
    "学习路径",
    "测试",
]

OUT_OF_DOMAIN_TERMS = [
    "天气",
    "天气预报",
    "火星",
    "股价",
    "股票",
    "彩票",
    "汇率",
    "房价",
    "旅游",
    "菜谱",
    "做饭",
    "电影",
    "明星",
    "nba",
    "足球",
    "高铁",
    "中考",
    "分数线",
]

VAGUE_PATTERNS = [
    "这个",
    "那个",
    "它",
    "该技术",
    "这些",
    "上面",
    "前面",
    "有什么用",
    "怎么用",
]


def _time_tool(
    state: AgentState,
    node: str,
    action: str,
    tool_name: str,
    input_summary: str,
    callback: Callable[[], Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    started_at = perf_counter()
    status = "success"
    metadata: Dict[str, Any] = {}

    try:
        output_summary, metadata = callback()
        return metadata
    except Exception as exc:
        status = "error"
        output_summary = str(exc)
        state.errors.append(f"{node}: {exc}")
        raise
    finally:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        state.add_trace(
            AgentTraceStep(
                node=node,
                action=action,
                tool_name=tool_name,
                input_summary=input_summary,
                output_summary=output_summary,
                status=status,
                elapsed_ms=elapsed_ms,
                metadata=metadata,
            )
        )


def _contains_any(text: str, keywords: List[str]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _is_single_chinese_char(text: str) -> bool:
    return len(text) == 1 and "\u4e00" <= text <= "\u9fff"


def _domain_term_matches(question: str, lower_question: str, term: str) -> bool:
    lower_term = term.lower()

    if not _is_single_chinese_char(term):
        return lower_term in lower_question or term in question

    stripped_question = question.strip(" ？?。！，,；;：:")
    single_char_patterns = [
        stripped_question == term,
        stripped_question.startswith(term),
        f"什么是{term}" in question,
        f"{term}是什么" in question,
        f"{term}的" in question,
        f"{term}有什么" in question,
    ]
    return any(single_char_patterns)


def is_project_domain_query(question: str) -> bool:
    if not _contains_any(question, PROJECT_DOMAIN_TERMS):
        return False

    return _contains_any(question, PROJECT_QUERY_CUES) or len(question.strip()) <= 24


def classify_query(question: str) -> Dict[str, Any]:
    stripped_question = question.strip()
    lower_question = stripped_question.lower()
    missing_check_terms = [
        "有没有",
        "是否包含",
        "资料中有没有",
        "知识库中有没有",
        "资料里有没有",
    ]
    missing_risk_terms = [
        "最新",
        "私人",
        "某",
        "全部",
        "内部",
        "实验数据",
        "电话号码",
        "联系方式",
        "融资",
        "政策",
        "招生",
        "录取",
        "投资",
        "gpu",
    ]

    if is_unsupported_fact_request(stripped_question):
        query_type = "missing"
        reason = "问题要求预测未来、获取隐私或查询外部实时事实，当前资料无法直接支持。"
    elif _contains_any(stripped_question, OUT_OF_DOMAIN_TERMS):
        query_type = "missing"
        reason = "问题包含明显超出当前教育资料知识库范围的主题。"
    elif any(term in stripped_question for term in missing_check_terms) and _contains_any(stripped_question, missing_risk_terms):
        query_type = "missing"
        reason = "问题是在询问当前知识库是否包含高风险或外部事实类资料。"
    elif is_project_domain_query(stripped_question):
        query_type = "project_concept"
        reason = "问题命中项目知识库术语，优先使用带来源上下文的混合检索。"
    elif any(word in stripped_question for word in ["区别", "对比", "相比", "不同", "差异"]) or " vs " in lower_question:
        query_type = "comparison"
        reason = "问题要求比较两个或多个概念。"
    elif any(word in stripped_question for word in ["怎么学", "如何学习", "学习建议", "学习路径", "入门", "提升"]):
        query_type = "learning_advice"
        reason = "问题关注学习路径或行动建议。"
    elif any(word in stripped_question for word in [
        "什么是",
        "解释",
        "定义",
        "原理",
        "机制",
        "作用",
        "为什么",
        "有哪些",
        "关系",
        "特点",
        "怎样分类",
        "如何分类",
        "说明了什么",
    ]):
        query_type = "concept"
        reason = "问题要求解释概念、原理或作用。"
    elif len(stripped_question) <= 10 or _contains_any(stripped_question, VAGUE_PATTERNS):
        query_type = "vague"
        reason = "问题较短或包含指代词，需要先改写成可检索查询。"
    else:
        query_type = "general"
        reason = "问题没有明显特殊约束，按普通知识库问答处理。"

    return {
        "query_type": query_type,
        "query_type_label": QUERY_TYPE_LABELS[query_type],
        "reason": reason,
    }


def select_strategy(query_type: str) -> str:
    if query_type == "project_concept":
        return "contextual_hybrid"

    if query_type == "general":
        return "contextual_hybrid"

    return "bm25_hybrid"


def rewrite_query_by_rule(question: str, query_type: str) -> str:
    lower_question = question.lower()

    matched_terms = [
        term
        for term in DOMAIN_TERMS
        if _domain_term_matches(question, lower_question, term)
    ]

    if matched_terms:
        subject = "、".join(dict.fromkeys(matched_terms[:4]))
    elif query_type == "comparison":
        subject = question
    else:
        subject = "当前教育领域知识库中的相关主题"

    if query_type == "learning_advice":
        return f"{subject} 的学习路径、应用场景和实践建议是什么？"

    if query_type == "comparison":
        return f"{subject} 的核心区别、适用场景和优缺点是什么？"

    if query_type == "missing":
        return f"当前教育资料知识库是否包含关于 {question} 的信息？"

    return f"{subject} 的概念、原理、作用和教材应用场景是什么？"


def _clean_rewrite_output(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    rewritten = lines[0]
    rewritten = re.sub(r"^[\-\*\d\.\s]+", "", rewritten).strip()
    rewritten = re.sub(
        r"^(改写后问题|改写问题|改写后查询|查询语句|检索查询|问题|Rewrite|Rewritten query)[:：]\s*",
        "",
        rewritten,
        flags=re.IGNORECASE,
    ).strip()
    rewritten = rewritten.strip("\"'“”‘’`")

    if len(rewritten) < 4:
        return ""

    if len(rewritten) > 120:
        rewritten = rewritten[:120].rstrip("，。；、 ")

    return rewritten


def build_rewrite_prompt(question: str, query_type: str, rule_query: str) -> str:
    return f"""
请把下面的用户问题改写成更适合 RAG 知识库检索的一句话。

要求：
1. 只输出改写后的检索问题；
2. 保留用户原意，不回答问题；
3. 如果问题含有“它、这个、这种”等指代词，请结合已补全的问题表达清楚主题；
4. 面向教育资料、初中科学教材、大模型应用开发资料检索；
5. 不要加入无法从原问题推断的新实体。

问题类型：{QUERY_TYPE_LABELS.get(query_type, query_type)}
原问题：{question}
规则改写参考：{rule_query}
""".strip()


def rewrite_query_with_metadata(
    question: str,
    query_type: str,
    use_llm: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    rule_query = rewrite_query_by_rule(question, query_type)

    if USE_MOCK or not use_llm:
        return rule_query, {
            "rewrite_strategy": "rule",
            "fallback_used": False,
            "reason": "Mock 模式或未启用 LLM Rewrite，使用规则改写。",
        }

    prompt = build_rewrite_prompt(
        question=question,
        query_type=query_type,
        rule_query=rule_query,
    )

    try:
        llm_output = call_llm(
            question=prompt,
            mode="query_rewrite",
        )
        rewritten_query = _clean_rewrite_output(llm_output)

        if rewritten_query and rewritten_query != question:
            return rewritten_query, {
                "rewrite_strategy": "llm",
                "fallback_used": False,
                "llm_output_preview": llm_output[:120],
            }

        return rule_query, {
            "rewrite_strategy": "rule_fallback",
            "fallback_used": True,
            "reason": "LLM Rewrite 输出为空或与原问题相同，回退到规则改写。",
            "llm_output_preview": llm_output[:120],
        }
    except Exception as exc:
        return rule_query, {
            "rewrite_strategy": "rule_fallback",
            "fallback_used": True,
            "reason": f"LLM Rewrite 调用失败，回退到规则改写：{str(exc)[:160]}",
        }


def rewrite_query(question: str, query_type: str) -> str:
    rewritten_query, _ = rewrite_query_with_metadata(question, query_type)
    return rewritten_query


def append_agent_log(result: Dict[str, Any]):
    AGENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if AGENT_LOG_FILE.exists():
        try:
            with AGENT_LOG_FILE.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logs = []
    else:
        logs = []

    logs.append(result)

    with AGENT_LOG_FILE.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def _retrieve_with_state(state: AgentState):
    chunks, retrieval_log = retrieve_chunks(
        question=state.current_query,
        top_k=state.top_k,
        retriever_mode=state.retriever_mode,
        candidate_k=state.candidate_k,
        use_rerank=state.use_rerank,
        context_mode=state.context_mode,
    )

    state.retrieved_chunks = chunks
    state.small_retrieved_chunks = retrieval_log.get("small_final_context", [])
    state.long_context = retrieval_log.get("long_context", {})
    state.retrieval_logs.append(retrieval_log)


def _generate_with_state(state: AgentState):
    if not state.context_sufficient:
        state.answer = build_no_context_answer(
            question=state.question,
            context_reason=state.context_reason,
        )
        state.sources = []
        state.claim_verification = {
            "enabled": True,
            "claim_count": 0,
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
            "claims": [],
            "status": "skipped_no_context",
        }
        return

    context = format_context(state.retrieved_chunks)
    rag_prompt = build_rag_prompt(
        question=state.resolved_question,
        context=context,
        guard_details=state.guard_details,
    )

    state.answer = call_llm(
        question=rag_prompt,
        mode="education",
    )
    state.sources = build_sources(state.retrieved_chunks)
    state.claim_verification = verify_answer_claims(
        answer=state.answer,
        chunks=state.retrieved_chunks,
    )


def run_rag_agent(
    question: str,
    top_k: int = 3,
    candidate_k: int = 10,
    max_rewrites: int = 1,
    use_rerank: bool = True,
    context_mode: str = "small_to_big",
    guard_mode: str = "v2",
    memory: Any = None,
    update_memory: bool = True,
) -> Dict[str, Any]:
    """
    Run the local Single-Agent RAG workflow.

    The Agent keeps the original RAG pipeline intact while adding query
    classification, retrieval strategy selection, query rewrite, context
    sufficiency judgement, trace logging, and controlled refusal.
    """

    memory_obj = ensure_memory(memory)

    state = AgentState(
        question=question.strip(),
        top_k=top_k,
        candidate_k=candidate_k,
        max_rewrites=max_rewrites,
        use_rerank=use_rerank,
        context_mode=context_mode,
        guard_mode=guard_mode,
    )

    if not state.question:
        raise ValueError("question 不能为空")

    if memory_obj is not None:
        def memory_step():
            memory_result = memory_obj.resolve_question(state.question)
            state.memory_used = memory_result["memory_used"]
            state.memory_reason = memory_result["memory_reason"]
            state.resolved_question = memory_result["resolved_question"]
            state.current_query = state.resolved_question
            state.memory_snapshot = memory_obj.to_dict()
            return (
                state.memory_reason,
                {
                    "memory_used": state.memory_used,
                    "resolved_question": state.resolved_question,
                    "current_topic": memory_obj.current_topic,
                    "current_topics": memory_obj.current_topics,
                },
            )

        _time_tool(
            state=state,
            node="resolve_memory",
            action="读取会话记忆并补全多轮追问",
            tool_name="conversation_memory",
            input_summary=state.question,
            callback=memory_step,
        )

    def classify_step():
        result = classify_query(state.resolved_question)
        state.query_type = result["query_type"]
        state.query_type_label = result["query_type_label"]
        state.query_reason = result["reason"]
        return (
            f"{state.query_type_label}：{state.query_reason}",
            result,
        )

    _time_tool(
        state=state,
        node="classify_query",
        action="识别用户问题类型",
        tool_name="rule_based_classifier",
        input_summary=state.resolved_question,
        callback=classify_step,
    )

    def select_step():
        state.retriever_mode = select_strategy(state.query_type)
        return (
            f"选择检索模式：{state.retriever_mode}",
            {
                "query_type": state.query_type,
                "retriever_mode": state.retriever_mode,
            },
        )

    _time_tool(
        state=state,
        node="select_strategy",
        action="根据问题类型选择检索策略",
        tool_name="strategy_router",
        input_summary=state.query_type_label,
        callback=select_step,
    )

    if (
        state.query_type == "vague"
        and not state.memory_used
        and state.rewrite_count < state.max_rewrites
    ):
        def pre_rewrite_step():
            rewritten_query, rewrite_meta = rewrite_query_with_metadata(
                state.resolved_question,
                state.query_type,
            )
            state.current_query = rewritten_query
            state.rewritten_queries.append(rewritten_query)
            state.rewrite_count += 1
            rewrite_step = {
                "rewrite_count": state.rewrite_count,
                "rewritten_query": rewritten_query,
                **rewrite_meta,
            }
            state.rewrite_steps.append(rewrite_step)
            return (
                f"改写为：{rewritten_query}",
                rewrite_step,
            )

        _time_tool(
            state=state,
            node="rewrite_query",
            action="将模糊问题改写为可检索查询",
            tool_name="query_rewriter",
            input_summary=state.resolved_question,
            callback=pre_rewrite_step,
        )

    while True:
        def retrieve_step():
            _retrieve_with_state(state)
            sources = build_sources(state.retrieved_chunks)
            source_names = [item["source"] for item in sources]
            return (
                f"召回 {len(state.retrieved_chunks)} 个回答片段，来源：{', '.join(source_names[:3])}",
                {
                    "retriever_mode": state.retriever_mode,
                    "context_mode": state.context_mode,
                    "current_query": state.current_query,
                    "long_context": state.long_context,
                    "small_retrieved_chunks": state.small_retrieved_chunks,
                    "retrieved_chunks": build_chunk_log_list(state.retrieved_chunks),
                },
            )

        _time_tool(
            state=state,
            node="retrieve_context",
            action="调用检索工具召回候选上下文",
            tool_name=state.retriever_mode,
            input_summary=state.current_query,
            callback=retrieve_step,
        )

        def judge_step():
            result = judge_context(
                original_question=state.resolved_question,
                current_query=state.current_query,
                chunks=state.retrieved_chunks,
                query_type=state.query_type,
                guard_mode=state.guard_mode,
            )
            state.context_sufficient = result["context_sufficient"]
            state.context_reason = result["reason"]
            state.context_coverage = round(float(result["coverage"]), 4)
            state.support_level = result.get("support_level", "supported" if state.context_sufficient else "unsupported")
            state.evidence_score = round(float(result.get("evidence_score", state.context_coverage)), 4)
            state.guard_details = result.get("guard_details", {})
            return (
                f"{'上下文充足' if state.context_sufficient else '上下文不足'}："
                f"{state.context_reason}",
                {
                    "context_sufficient": state.context_sufficient,
                    "context_coverage": state.context_coverage,
                    "context_reason": state.context_reason,
                    "support_level": state.support_level,
                    "evidence_score": state.evidence_score,
                    "guard_mode": state.guard_mode,
                    "guard_details": state.guard_details,
                },
            )

        _time_tool(
            state=state,
            node="judge_context",
            action="判断检索上下文是否足够回答",
            tool_name="context_judge",
            input_summary=state.current_query,
            callback=judge_step,
        )

        if (
            state.context_sufficient
            or state.query_type == "missing"
            or state.rewrite_count >= state.max_rewrites
        ):
            break

        def retry_rewrite_step():
            rewritten_query, rewrite_meta = rewrite_query_with_metadata(
                state.question,
                state.query_type,
            )

            if rewritten_query == state.current_query:
                rewritten_query = (
                    f"{state.question} 请结合当前教育领域知识库中的大模型技术资料或"
                    "初中科学教材资料回答。"
                )
                rewrite_meta = {
                    **rewrite_meta,
                    "fallback_used": True,
                    "reason": "改写结果与当前查询重复，追加知识库范围约束。",
                }

            state.current_query = rewritten_query
            state.rewritten_queries.append(rewritten_query)
            state.rewrite_count += 1
            rewrite_step = {
                "rewrite_count": state.rewrite_count,
                "rewritten_query": rewritten_query,
                **rewrite_meta,
            }
            state.rewrite_steps.append(rewrite_step)
            return (
                f"二次检索查询：{rewritten_query}",
                rewrite_step,
            )

        _time_tool(
            state=state,
            node="rewrite_query",
            action="上下文不足，改写查询并准备二次检索",
            tool_name="query_rewriter",
            input_summary=state.question,
            callback=retry_rewrite_step,
        )

    def generate_step():
        _generate_with_state(state)
        return (
            f"生成回答，引用来源数：{len(state.sources)}",
            {
                "answer_length": len(state.answer),
                "source_count": len(state.sources),
                "context_sufficient": state.context_sufficient,
            },
        )

    _time_tool(
        state=state,
        node="generate_answer",
        action="基于上下文生成回答或执行拒答",
        tool_name="rag_answer_generator" if state.context_sufficient else "rule_based_refusal",
        input_summary=state.resolved_question,
        callback=generate_step,
    )

    if memory_obj is not None and update_memory:
        def update_memory_step():
            memory_obj.update_from_result(
                question=state.question,
                resolved_question=state.resolved_question,
                answer=state.answer,
                sources=state.sources,
                query_type=state.query_type,
                retriever_mode=state.retriever_mode,
                rewritten_queries=state.rewritten_queries,
            )
            state.memory_snapshot = memory_obj.to_dict()
            return (
                f"记忆已更新，当前主题：{memory_obj.current_topic or '无'}",
                state.memory_snapshot,
            )

        _time_tool(
            state=state,
            node="update_memory",
            action="写入本轮问题、主题、摘要和来源",
            tool_name="conversation_memory",
            input_summary=state.resolved_question,
            callback=update_memory_step,
        )

    def finalize_step():
        return (
            "整理最终回答、引用来源和 Agent 执行轨迹。",
            {
                "trace_step_count": len(state.agent_trace),
                "rewritten_queries": state.rewritten_queries,
                "rewrite_steps": state.rewrite_steps,
                "memory_used": state.memory_used,
            },
        )

    _time_tool(
        state=state,
        node="finalize_response",
        action="统一输出 Agent 结果",
        tool_name="response_builder",
        input_summary=state.question,
        callback=finalize_step,
    )

    result = state.to_result()

    result["agent_log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["retrieval_logs"] = state.retrieval_logs

    append_agent_log(result)

    return result


if __name__ == "__main__":
    demo_result = run_rag_agent("什么是 RAG？")

    print("问题类型：", demo_result["query_type_label"])
    print("检索模式：", demo_result["retriever_mode"])
    print("上下文充足：", demo_result["context_sufficient"])
    print("回答：")
    print(demo_result["answer"])
