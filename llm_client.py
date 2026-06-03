import requests

from config import API_KEY, BASE_URL, MODEL_NAME, USE_MOCK
from prompt_templates import get_prompt


def call_llm(question: str, mode: str = "general") -> str:
    """
    调用大模型或返回模拟回答。

    参数：
        question: 用户输入的问题；
        mode: 问答模式，例如 general、education、paper_summary。

    当前支持：
        1. USE_MOCK=true：返回 Mock 模拟回答；
        2. USE_MOCK=false：调用硅基流动 DeepSeek API。
    """

    system_prompt = get_prompt(mode)

    if USE_MOCK:
        return generate_mock_answer(question, mode, system_prompt)

    if not API_KEY:
        raise ValueError("没有检测到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    return call_real_llm_api(
        question=question,
        system_prompt=system_prompt,
    )


def call_real_llm_api(question: str, system_prompt: str) -> str:
    """
    调用 OpenAI 兼容格式的大模型 API。

    当前用于：
        硅基流动平台的 DeepSeek 系列模型。
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "stream": False,
        "temperature": 0.7,
    }

    response = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API 调用失败，状态码：{response.status_code}，返回内容：{response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 返回格式异常：{data}") from e


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


if __name__ == "__main__":
    test_question = "请用通俗语言解释什么是 RAG。"

    answer = call_llm(
        question=test_question,
        mode="education",
    )

    print(answer)