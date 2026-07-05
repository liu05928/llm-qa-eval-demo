import re
from typing import Optional

import requests

from config import (
    API_KEY,
    BASE_URL,
    GENERATION_BACKEND,
    LOCAL_SFT_API_KEY,
    LOCAL_SFT_BASE_URL,
    LOCAL_SFT_MAX_TOKENS,
    LOCAL_SFT_MODEL,
    LOCAL_SFT_REPETITION_PENALTY,
    LOCAL_SFT_TEMPERATURE,
    LOCAL_SFT_TIMEOUT_SECONDS,
    LOCAL_SFT_TOP_P,
    MODEL_NAME,
    USE_MOCK,
)
from prompt_templates import get_prompt


VALID_GENERATION_BACKENDS = {"mock", "api", "local_sft"}


def normalize_generation_backend(generation_backend: Optional[str] = None) -> str:
    backend = (generation_backend or GENERATION_BACKEND or "").strip().lower()

    if not backend:
        backend = "mock" if USE_MOCK else "api"

    if backend not in VALID_GENERATION_BACKENDS:
        valid = ", ".join(sorted(VALID_GENERATION_BACKENDS))
        raise ValueError(f"不支持的 generation_backend：{backend}。可选值：{valid}")

    return backend


def get_llm_runtime_info(generation_backend: Optional[str] = None) -> dict:
    backend = normalize_generation_backend(generation_backend)

    if backend == "local_sft":
        model = LOCAL_SFT_MODEL
    elif backend == "api":
        model = MODEL_NAME
    else:
        model = "mock"

    return {
        "generation_backend": backend,
        "generator_model": model,
        "mock": backend == "mock",
    }


def call_llm(
    question: str,
    mode: str = "general",
    generation_backend: Optional[str] = None,
) -> str:
    """
    调用大模型或返回模拟回答。

    参数：
        question: 用户输入的问题；
        mode: 问答模式，例如 general、education、paper_summary。
        generation_backend: mock、api 或 local_sft。

    当前支持：
        1. mock：返回 Mock 模拟回答；
        2. api：调用 SiliconFlow / DeepSeek API；
        3. local_sft：调用 OpenAI-compatible 微调模型服务。
    """

    system_prompt = get_prompt(mode)
    backend = normalize_generation_backend(generation_backend)

    if backend == "mock":
        return generate_mock_answer(question, mode, system_prompt)

    if backend == "local_sft":
        if not LOCAL_SFT_BASE_URL:
            raise ValueError("LOCAL_SFT_BASE_URL 未配置，无法调用本地/云端 SFT 模型。")

        return call_openai_compatible_api(
            question=question,
            system_prompt=system_prompt,
            api_key=LOCAL_SFT_API_KEY,
            base_url=LOCAL_SFT_BASE_URL,
            model_name=LOCAL_SFT_MODEL,
            backend_name="local_sft",
            temperature=LOCAL_SFT_TEMPERATURE,
            top_p=LOCAL_SFT_TOP_P,
            max_tokens=LOCAL_SFT_MAX_TOKENS,
            timeout=LOCAL_SFT_TIMEOUT_SECONDS,
            repetition_penalty=LOCAL_SFT_REPETITION_PENALTY,
        )

    if not API_KEY:
        raise ValueError("没有检测到 SILICONFLOW_API_KEY / DEEPSEEK_API_KEY，请检查 .env 文件。")

    return call_openai_compatible_api(
        question=question,
        system_prompt=system_prompt,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        backend_name="api",
    )


def call_openai_compatible_api(
    question: str,
    system_prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
    backend_name: str,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
    repetition_penalty: Optional[float] = None,
) -> str:
    """
    调用 OpenAI 兼容格式的大模型 API。
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'EMPTY'}",
    }

    payload = {
        "model": model_name,
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
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty

    response = requests.post(
        base_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"{backend_name} API 调用失败，状态码：{response.status_code}，返回内容：{response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 返回格式异常：{data}") from e


def call_real_llm_api(question: str, system_prompt: str) -> str:
    """
    兼容旧调用：调用 SiliconFlow / DeepSeek API。
    """

    return call_openai_compatible_api(
        question=question,
        system_prompt=system_prompt,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        backend_name="api",
    )


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

    if mode == "education" and "用户问题：" in question and "参考资料：" in question:
        user_question = question.split("用户问题：", 1)[1].split("参考资料：", 1)[0].strip()

        if "证据判断：" in user_question:
            user_question = user_question.split("证据判断：", 1)[0].strip()

        context = question.split("参考资料：", 1)[1].strip()
        source_match = re.search(r"\[来源：([^，\]]+)", context)
        source = source_match.group(1) if source_match else "mock_source"
        context_preview = re.sub(r"\s+", " ", context)[:180]

        return (
            "关键词：资料依据、知识库、学习理解\n\n"
            f"教材依据：{context_preview}\n\n"
            f"回答：根据当前检索资料，可以围绕“{user_question}”进行解释。"
            "资料中的相关片段提供了回答该问题的依据，因此本回答只概括这些已检索内容，"
            "不补充外部事实。\n\n"
            "学习建议：复习时先定位资料中的概念、现象或公式，再用自己的话复述其因果关系。\n\n"
            f"参考来源：{source}"
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

    if mode == "query_rewrite":
        marker = "原问题："
        if marker in question:
            raw_question = question.split(marker, 1)[1].splitlines()[0].strip()
        else:
            raw_question = question.strip().splitlines()[-1]

        return raw_question

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
