# 运行与复现说明

本文档记录项目的本地运行、知识库构建、SFT 数据生成、训练配置、评测命令和 `local_sft` 接入方式。README 只保留最小入口，详细步骤集中放在这里。

## 1. 环境准备

建议使用 Python 3.11。

```bash
pip install -r requirements.txt
cp .env.example .env
```

默认 `USE_MOCK=true`、`GENERATION_BACKEND=mock`，可以在不调用外部模型服务的情况下完成本地演示和大部分链路检查。需要真实模型时，在 `.env` 中改为对应后端并配置供应商凭据或 OpenAI-compatible 本地服务地址。

## 2. 构建知识库

构建知识库会读取 `data/raw_docs/`，生成 small/big chunks，并建立 ChromaDB 向量索引。

```bash
USE_MOCK=true python3 build_kb.py
```

常见输出：

- `data/chunks/chunks.json`：small chunks，用于召回。
- `data/chunks/big_chunks.json`：big chunks，用于 Small-to-Big 回答上下文。
- `vector_db/`：本地向量库。
- `eval_results/build_kb_summary.json`：构建摘要。

如需在构建后同时跑科学教材 long-text 对比评测：

```bash
USE_MOCK=true python3 build_kb.py --run-eval
```

## 3. 启动服务

启动 FastAPI：

```bash
uvicorn app:app --reload
```

服务启动后可访问：

- `GET /health`：服务健康检查。
- `POST /chat`：基础问答。
- `POST /rag_chat`：检索增强问答。
- `POST /agent/chat`：Workflow/Agent 问答。

启动 Streamlit：

```bash
streamlit run web_demo.py
```

页面包含 Agent 问答、RAG 问答、评测结果和日志查看。

## 4. Docker Compose

Docker Compose 会同时启动 FastAPI、Streamlit 和 Redis。

```bash
docker compose up --build
```

默认端口：

- FastAPI：`http://localhost:8000`
- Streamlit：`http://localhost:8501`
- Redis：`localhost:6379`

Compose 会挂载 `data/`、`eval_results/`、`logs/` 和 `vector_db/`，便于本地结果复用。

## 5. SFT 数据生成

首版小规模数据：

```bash
python3 sft_dataset_builder.py
```

v2 规模化数据：

```bash
python3 sft_dataset_builder_v2.py
```

v2.1 拒答增强与反重复数据：

```bash
python3 sft_dataset_builder_v21.py
```

当前 v2.1 数据集统计：

- 总量：3000
- train/dev/test：2400/300/300
- 数据类型：`grounded_qa`、`citation_qa`、`compare_or_reasoning`、`query_rewrite`、`refusal`、`anti_repetition`
- hard-refusal 专项评测题：100 条，不进入训练集
- 固定质量评测题：50 条，不进入训练集

## 6. LLaMA Factory 训练配置

训练配置位于 `training/llamafactory/`：

- `qwen25_3b_qlora_sft.yaml`
- `qwen25_3b_qlora_sft_v2.yaml`
- `qwen25_3b_qlora_sft_v21.yaml`
- `qwen25_3b_qlora_api.yaml`
- `qwen25_3b_qlora_api_v2.yaml`
- `qwen25_3b_qlora_api_v21.yaml`

推荐训练路线：

- 12GB-16GB GPU：优先 QLoRA。
- 24GB GPU：可尝试 LoRA。
- 大模型权重、adapter 权重和 checkpoint 不进入仓库；训练摘要与轻量日志如需保留，建议放在 ignored 的本地归档目录中。

v2.1 已完成训练验证。公开复现说明见 `docs/training_reproduction.md`，训练配置见 `training/llamafactory/`。

## 7. 接入 local_sft 后端

当微调模型以 OpenAI-compatible chat completions 服务启动后，在 `.env` 中切换：

```text
USE_MOCK=false
GENERATION_BACKEND=local_sft
```

同时将 `.env.example` 中的 local_sft 模型名、服务地址、采样参数同步为实际服务配置。完成后，`/chat`、`/rag_chat` 和 `/agent/chat` 都可以通过 `generation_backend=local_sft` 调用微调模型。

可先检查模型服务是否可用：

```bash
curl -sS http://127.0.0.1:8001/v1/models
```

## 8. 评测命令

SFT 生成质量评测：

```bash
python3 sft_generation_eval_runner.py --questions data/sft_v21/quality_eval_questions.json --output-dir eval_results/sft_generation_v21 --run-name v21_qlora_fixed50
```

hard-refusal 专项评测：

```bash
python3 sft_generation_eval_runner.py --questions data/sft_v21/hard_refusal_eval_questions.json --output-dir eval_results/sft_generation_v21 --run-name v21_qlora_hard_refusal100
```

RAG/Workflow guard 端到端评测：

```bash
python3 rag_workflow_guard_eval_runner.py --backend mock --run-name smoke_mock_guard
```

如果本地已经接入 `local_sft` 服务，可将 `--backend` 改为 `local_sft`。

主要报告位置：

- `eval_results/sft_generation_v21/comparison_report.md`
- `eval_results/rag_workflow_guard/*/summary.json`
- `eval_results/science_small_to_big_report.md`
- `eval_results/science_failure_cases.md`

## 9. 结果解释口径

v2.1 的 hard-refusal 数据补丁显著提升了专项拒答正确率，并消除了当前评测中观察到的重复段落问题。但固定题中的开放型无依据问题仍可能触发模型强答，因此项目同时在 RAG/Workflow 层使用 `context_guard.py` 做上下文充分性判断。

对外描述时建议使用“降低无依据回答风险”“提升拒答稳定性”“通过证据和 Workflow 控制约束回答”，不要把单次评测结果扩大为对所有开放问题的绝对保证。
