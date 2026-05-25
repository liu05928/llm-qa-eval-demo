# 大模型问答与评测接口系统

## 一、项目简介

本项目是一个面向大模型应用开发实践的问答与评测接口系统，基于 Python 和 FastAPI 构建。

当前版本采用 Mock 模式运行，已实现命令行问答、FastAPI 问答接口、Prompt 模板管理、多模式问答、问答日志保存、测试问题集构建和简单关键词命中评测流程，用于跑通大模型应用开发中的基础工程链路。后续可接入 DeepSeek、通义千问或 OpenAI 等真实大模型 API，并扩展为教育场景 RAG 知识库问答系统。

## 二、项目背景

本项目围绕大模型应用开发、AI Agent、RAG 和大模型评测等方向展开，用于训练大模型应用项目的基础工程能力。项目从最基础的大模型调用流程出发，逐步完成命令行问答、接口封装、Prompt 模板管理和自动评测模块，重点训练以下能力：

1. Python 项目结构组织；
2. FastAPI 后端接口开发；
3. Prompt 模板设计；
4. 多模式问答接口设计；
5. JSON 日志保存；
6. 测试问题集构建；
7. 简单关键词命中评测；
8. 为后续真实大模型 API、RAG 和 Agent 功能扩展做准备。

## 三、技术栈

### 当前已使用

- Python
- FastAPI
- Uvicorn
- Pydantic
- python-dotenv
- JSON
- pathlib
- Prompt Engineering
- Mock 模拟模式
- 简单关键词评测

### 后续计划扩展

- Markdown / TXT 文档读取
- 文本切分
- Embedding 向量化
- Chroma 向量数据库
- RAG 检索增强生成
- Streamlit 简易演示页面

## 四、当前功能

1. 支持命令行问答；
2. 支持 FastAPI 后端接口；
3. 支持 POST `/chat` 问答接口；
4. 支持 GET `/health` 健康检查接口；
5. 支持 GET `/modes` 查看所有问答模式；
6. 支持 Prompt 模板管理；
7. 支持通过 `mode` 参数选择不同问答模式；
8. 当前支持 `general`、`education`、`paper_summary` 三种模式；
9. 支持将问题、回答、时间、模型名称、运行模式保存到本地日志；
10. 支持测试问题集 `data/test_questions.json`；
11. 支持基于关键词命中的简单自动评测；
12. 支持 POST `/evaluate` 运行评测；
13. 支持 GET `/eval-results` 查看最近一次评测结果。
14. 已准备教育资料 RAG 示例文档，后续用于文档读取、文本切分和向量检索。

## 五、项目结构

```text
llm-qa-eval-demo/
├── app.py                  # FastAPI 接口入口
├── main.py                 # 命令行问答入口
├── config.py               # 项目配置管理
├── llm_client.py           # 模型调用与 Mock 回答逻辑
├── chat_logger.py          # 问答日志保存
├── prompt_templates.py     # Prompt 模板管理
├── evaluator.py            # 简单评测模块
├── data/
│   ├── raw_docs/           # RAG 示例知识库文档
│   │   ├── rag_intro.md
│   │   ├── agent_intro.md
│   │   ├── prompt_engineering.md
│   │   └── education_ai.md
│   ├── chunks/             # 后续保存文本切分结果
│   └── test_questions.json # 关键词评测测试问题集
├── logs/
│   └── chat_log.json       # 本地运行生成，已加入 .gitignore
├── results/
│   └── eval_results.json   # 本地评测生成，已加入 .gitignore
├── vector_db/              # 后续保存本地向量数据库，已加入 .gitignore
├── assets/                 # 项目截图
├── .env                    # 环境变量文件，已加入 .gitignore
├── .gitignore
├── README.md
└── requirements.txt
```
## 六、RAG 扩展规划

当前项目已经完成基础问答接口、Prompt 模板管理、日志记录和关键词命中评测流程。后续将在现有系统基础上扩展为教育场景 RAG 知识库问答系统。

RAG 的核心思想是：在大模型生成答案之前，先从本地知识库中检索与用户问题相关的文本片段，再将检索内容和用户问题一起放入 Prompt 中，让模型基于资料生成回答。

相比普通问答系统，RAG 系统可以让回答更依赖指定资料，并返回引用来源，从而提高回答的可追溯性，减少模型幻觉。

计划新增模块包括：

- `document_loader.py`：读取本地 Markdown 或 TXT 文档；
- `text_splitter.py`：将长文档切分为多个文本块；
- `embedding_client.py`：将文本转换为向量表示；
- `vector_store.py`：存储和检索文本向量；
- `rag_pipeline.py`：串联“检索 + 生成”的 RAG 主流程；
- `web_demo.py`：提供简单的 Streamlit 演示页面。

计划实现流程：

```text
本地教育资料
↓
文档读取
↓
文本切分
↓
向量化
↓
向量检索
↓
构造 RAG Prompt
↓
生成回答
↓
返回引用来源
↓
记录日志
↓
进行简单评测