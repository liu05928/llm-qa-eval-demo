from typing import Dict, Optional


PROMPT_TEMPLATES: Dict[str, str] = {
    "general": (
        "你是一个严谨、清晰的大模型助手。"
        "请直接回答用户问题，语言简洁，结构清楚。"
    ),

    "education": (
        "你是一个适合科教场景的智能助教。"
        "请用适合初学者理解的方式回答问题。"
        "回答时尽量包含：概念解释、简单例子、学习建议。"
    ),

    "paper_summary": (
        "你是一个科研论文阅读助手。"
        "请帮助用户理解论文或学术内容。"
        "回答时尽量包含：研究背景、核心问题、方法思路、可能贡献和局限。"
    ),

    "query_rewrite": (
        "你是一个 RAG 检索查询改写器。"
        "你的任务是把用户问题改写成更适合知识库检索的一句话。"
        "只输出改写后的问题，不要解释，不要回答问题。"
        "改写时保留原问题的核心意图，补全必要主题词，避免加入无法从问题推断的新事实。"
    )
}


PUBLIC_MODES = ["general", "education", "paper_summary"]


def get_prompt(mode: str) -> str:
    """
    根据 mode 获取对应的 Prompt 模板。

    参数：
        mode: 用户选择的模式，例如 general、education、paper_summary。

    返回：
        对应模式的系统提示词。

    如果 mode 不存在，就抛出 ValueError。
    """

    if mode not in PROMPT_TEMPLATES:
        valid_modes = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"不支持的 mode：{mode}。可选 mode 包括：{valid_modes}")

    return PROMPT_TEMPLATES[mode]


def get_available_modes(include_internal: bool = False) -> list:
    """
    返回当前支持的所有模式。
    """

    if include_internal:
        return list(PROMPT_TEMPLATES.keys())

    return PUBLIC_MODES.copy()


def build_rag_prompt(question: str, context: str, guard_details: Optional[dict] = None) -> str:
    """
    构造 RAG 问答 Prompt。

    参数：
    - question: 用户问题
    - context: 向量检索得到的参考资料

    返回：
    - 拼接好的 RAG Prompt
    """

    guard_text = ""

    if guard_details:
        support_level = guard_details.get("support_level", "unknown")
        evidence_score = guard_details.get("evidence_score", "")
        missing_terms = "、".join(guard_details.get("missing_terms", [])[:12])
        guard_text = f"""

证据判断：
- support_level: {support_level}
- evidence_score: {evidence_score}
- missing_terms: {missing_terms or "无"}
""".rstrip()

    prompt = f"""
你是一名教育领域的大模型知识库问答助手。
请严格根据下面提供的参考资料回答用户问题。

要求：
1. 只依据参考资料回答，不补充参考资料外的事实、数字、政策、时间或预测；
2. 先判断问题中的每个关键点是否被资料直接支持，只回答被支持的部分；
3. 如果关键点没有被资料直接支持，请明确说明“资料中未提及”，不要把低相关片段当作答案；
4. 回答中的每个核心结论都要能在参考资料中找到对应依据；
5. 回答要清晰、分点说明，最后给出参考来源；
6. 如果证据判断显示 partial 或 unsupported，请优先拒答或只说明资料边界。

用户问题：
{question}
{guard_text}

参考资料：
{context}
"""
    return prompt.strip()
