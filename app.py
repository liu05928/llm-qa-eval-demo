from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import rag_answer

from config import MODEL, USE_MOCK
from llm_client import call_llm
from chat_logger import save_log
from prompt_templates import get_available_modes
from evaluator import run_evaluation, load_eval_results


app = FastAPI(
    title="大模型问答接口系统",
    description="基于 FastAPI 的大模型问答 Demo，支持 Prompt 模板、多模式问答和简单评测。",
    version="0.3.0"
)


class ChatRequest(BaseModel):
    """
    用户请求体。

    question: 用户问题
    mode: 问答模式
    """

    question: str
    mode: Literal["general", "education", "paper_summary"] = "general"

class RagChatRequest(BaseModel):
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    """
    系统返回体。
    """

    question: str
    answer: str
    mode: str
    model: str
    mock: bool


@app.get("/")
def root():
    """
    根路径接口，用来检查服务是否启动成功。
    """

    return {
        "message": "大模型问答接口系统已启动",
        "docs": "请访问 /docs 查看接口文档",
        "available_modes": get_available_modes()
    }


@app.get("/health")
def health_check():
    """
    健康检查接口。
    """

    return {
        "status": "ok",
        "model": MODEL,
        "mock": USE_MOCK,
        "available_modes": get_available_modes()
    }


@app.get("/modes")
def list_modes():
    """
    查看当前支持的所有问答模式。
    """

    return {
        "available_modes": get_available_modes(),
        "description": {
            "general": "通用问答模式",
            "education": "教育解释模式",
            "paper_summary": "论文总结模式"
        }
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    问答接口。

    输入示例：
    {
        "question": "什么是 RAG？",
        "mode": "education"
    }
    """

    question = request.question.strip()
    mode = request.mode

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        answer = call_llm(question, mode=mode)
        save_log(question, answer, mode=mode)

        return ChatResponse(
            question=question,
            answer=answer,
            mode=mode,
            model=MODEL,
            mock=USE_MOCK
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_chat")
def rag_chat(request: RagChatRequest):
    """
    RAG 知识库问答接口。

    输入：
    - question: 用户问题
    - top_k: 检索返回的文本块数量

    输出：
    - question: 原始问题
    - answer: 模型回答
    - sources: 引用来源
    - retrieved_chunks: 检索到的文本块
    """

    result = rag_answer(
        question=request.question,
        top_k=request.top_k,
    )

    return result

@app.post("/evaluate")
def evaluate():
    """
    运行一轮自动评测。

    系统会读取 data/test_questions.json，
    对每个测试问题调用 call_llm，
    然后根据 expected_keywords 计算关键词命中得分。
    """

    try:
        summary = run_evaluation()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval-results")
def get_eval_results():
    """
    查看最近一次评测结果。
    """

    return load_eval_results()