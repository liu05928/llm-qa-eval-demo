# 截图清单

本文档用于整理项目展示材料时检查截图是否齐全。截图文件建议放在 `assets/` 或外部材料目录中，README 中只保留必要的少量截图或链接。

## 服务与页面

- FastAPI `/health` 返回服务状态、生成后端和运行时控制状态。
- FastAPI `/docs` 展示 `/chat`、`/rag_chat`、`/agent/chat` 等接口。
- Streamlit Agent 问答页面，包含问题类型、检索策略、上下文判断、记忆状态和回答。
- Streamlit RAG 问答页面，包含检索模式、模型回答、引用来源和检索片段。
- Streamlit 评测结果页面，展示来源命中率、关键词命中率、引用完整率和无资料拒答率。

## 可信回答链路

- 一个有教材依据的问题：回答中包含来源引用。
- 一个资料不足的问题：系统返回无来源拒答。
- Query Rewrite 展示：口语化问题被改写为更适合检索的问题。
- Small-to-Big 展示：small chunk 触发父级 big chunk 上下文。
- Agent 工具调用轨迹：展示 Workflow 节点和耗时。

## 训练与评测

- v2.1 训练摘要，展示训练配置、数据规模和关键 loss。
- SFT 生成质量对比表，展示 v2 与 v2.1 fixed50 指标。
- hard-refusal100 评测摘要，展示拒答正确率。
- RAG/Workflow guard smoke test 摘要，展示三条路径通过率。
- 失败样例分析报告，展示后续数据补丁或 guard 规则的依据。

## 工程化交付

- Docker Compose 启动后的 API、Web、Redis 三个服务状态。
- 项目目录结构，突出 `data/`、`eval_results/`、`training/llamafactory/` 和 `docs/`。
- `.env.example` 片段，展示 mock、API 和 local_sft 三种后端的可配置性。
- 训练配置目录，展示 LoRA/QLoRA 和 API 配置文件。

## 截图规范

- 不展示真实密钥、个人本地绝对路径或临时云服务信息。
- 命令行截图只保留关键成功信息，避免包含过长日志。
- 评测截图需要同时保留指标名称和样本规模，避免只截单个百分比。
- 对外材料中优先展示系统能力和真实结果，避免使用夸大措辞。
