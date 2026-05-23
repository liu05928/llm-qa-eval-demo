from typing import Dict


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
    )
}


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


def get_available_modes() -> list:
    """
    返回当前支持的所有模式。
    """

    return list(PROMPT_TEMPLATES.keys())