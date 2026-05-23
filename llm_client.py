from config import API_KEY, USE_MOCK
from prompt_templates import get_prompt


def call_llm(question: str, mode: str = "general") -> str:
    """
    调用大模型或返回模拟回答。

    参数：
        question: 用户输入的问题；
        mode: 问答模式，例如 general、education、paper_summary。

    当前阶段：
        使用 Mock 模式，根据不同 mode 返回不同风格的模拟回答。

    后续阶段：
        接入真实大模型 API 时，会把 system_prompt 和 question 一起发送给模型。
    """

    system_prompt = get_prompt(mode)

    if USE_MOCK:
        return generate_mock_answer(question, mode, system_prompt)

    if not API_KEY:
        raise ValueError("没有检测到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    return "这里未来会接入真实大模型 API。"


def generate_mock_answer(question: str, mode: str, system_prompt: str) -> str:
    """
    根据不同 mode 生成模拟回答。

    这样即使没有真实 API，
    我们也能先测试 Prompt 模板和 mode 参数是否生效。
    """

    if mode == "general":
        return (
            f"【通用问答模式】\n"
            f"你的问题是：{question}\n\n"
            f"这是一个模拟回答。真实接入大模型 API 后，系统会根据通用问答提示词生成更完整的回答。"
        )

    if mode == "education":
        return (
            f"【教育解释模式】\n"
            f"你的问题是：{question}\n\n"
            f"概念解释：这里会用适合初学者理解的方式解释这个问题。\n"
            f"简单例子：这里会给出一个贴近日常或学习场景的例子。\n"
            f"学习建议：这里会给出后续学习建议。\n\n"
            f"当前为 Mock 模式，后续接入真实 API 后会生成更自然的教育解释。"
        )

    if mode == "paper_summary":
        return (
            f"【论文总结模式】\n"
            f"你的输入是：{question}\n\n"
            f"研究背景：这里会总结相关研究背景。\n"
            f"核心问题：这里会提炼论文试图解决的问题。\n"
            f"方法思路：这里会概括论文方法。\n"
            f"贡献与局限：这里会分析可能贡献和不足。\n\n"
            f"当前为 Mock 模式，后续接入真实 API 后会根据论文内容生成摘要。"
        )

    return (
        f"【未知模式】\n"
        f"你的问题是：{question}\n"
        f"当前 mode={mode}，请检查 Prompt 模板配置。"
    )