# 向量数据库在 RAG 中的作用

向量数据库用于存储和检索 Embedding 向量，是 RAG 系统的核心基础设施之一。普通数据库更擅长按字段精确查询，例如按标题、时间或编号查询；向量数据库更擅长按语义相似度查询，例如用户问“RAG 检索流程是什么”，系统可以找到包含“问题向量化、相似度检索、Top-K 召回、上下文拼接”的资料片段。

当前项目使用 ChromaDB 作为向量数据库。构建索引时，系统会读取 `data/chunks/chunks.json`，取出每个 chunk 的 content 生成 Embedding，并把向量、文本、chunk_id 和 source 写入 Chroma collection。ChromaDB 会把向量库持久化到 `vector_db/` 目录，因此构建完成后，后续问答可以直接加载已经保存的向量索引。

向量数据库检索的基本流程是：先把用户问题转换成查询向量，再在 collection 中查找距离最近的若干文本向量。这个返回数量通常叫 Top-K。Top-K 越小，返回内容越精简，但可能漏掉有用资料；Top-K 越大，召回更充分，但也可能带来噪声，增加大模型上下文压力。当前项目默认会根据配置或调用参数返回若干条最相关 chunks。

ChromaDB 返回的检索结果中包含 distance。distance 表示查询向量和文档向量之间的距离，通常距离越小，语义越相关。为了便于展示和融合，项目会把 distance 转换为 dense_score，例如使用 `1 / (1 + distance)` 得到一个越大越相关的分数。

一个向量数据库中的每条记录通常包含：

- id：文本块唯一标识，例如 chunk_id；
- document：原始文本块内容；
- embedding：由 Embedding 模型生成的向量；
- metadata：来源文件、chunk_id 或其他辅助字段；
- distance 或 score：查询时返回的相关性信息。

向量数据库不是知识库的全部。它只保存向量和可检索文本，真正的知识组织还依赖文档质量、切分策略、元数据和检索流程。如果 raw_docs 中加入了新的 Markdown 文件，需要先运行文本切分脚本生成新的 chunks，再删除旧的 `vector_db/` 并重新构建索引。这样新增资料才会进入 ChromaDB，被后续 RAG 问答检索到。

