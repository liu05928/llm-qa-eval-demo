# 大模型问答与评测接口系统

## 一、项目简介

本项目是一个面向大模型应用开发学习的问答与评测接口系统，基于 Python 和 FastAPI 构建。

当前版本采用 Mock 模式运行，已实现命令行问答、FastAPI 问答接口、Prompt 模板、多模式问答、问答日志保存、测试问题集和简单关键词评测流程。后续可接入 DeepSeek、通义千问或 OpenAI 等真实大模型 API，并进一步扩展为教育场景 RAG 知识库问答系统。

## 二、项目背景

本项目用于准备大模型应用开发、AI Agent、RAG 和大模型评测相关实习。项目从最基础的大模型调用流程出发，逐步完成命令行问答、接口封装、Prompt 模板管理和自动评测模块，重点训练以下能力：

1. Python 项目结构组织；
2. FastAPI 后端接口开发；
3. Prompt 模板设计；
4. 多模式问答接口设计；
5. JSON 日志保存；
6. 测试问题集构建；
7. 简单关键词命中评测；
8. 为后续真实大模型 API、RAG 和 Agent 功能扩展做准备。

## 三、技术栈

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

## 五、项目结构

```text
llm-demo/
├── app.py
├── main.py
├── config.py
├── llm_client.py
├── chat_logger.py
├── prompt_templates.py
├── evaluator.py
├── data/
│   └── test_questions.json
├── logs/
│   └── chat_log.json
├── results/
│   └── eval_results.json
├── .env
├── .gitignore
├── README.md
└── requirements.txt