# Rerank 重排序入门

Rerank 是 RAG 检索流程中的二阶段排序方法。第一阶段通常由向量检索或 Hybrid Search 负责召回候选 chunks，目标是尽量把可能相关的资料找回来。第二阶段由 Rerank 模型对“用户问题”和“候选文本块”逐对判断相关性，再按相关性分数重新排序。Rerank 的重点是提高最终上下文质量。

当前项目使用 BAAI/bge-reranker-v2-m3 作为重排序模型。系统会先通过 Hybrid Search 获得 candidate_k 个候选 chunks，再把每个 chunk 的 content 和用户问题一起发送给 Rerank 客户端。模型返回 relevance_score，表示该候选文本对当前问题的相关程度。项目会把这个分数保存为 rerank_score，并记录 rerank_model，方便在日志和 Streamlit 页面中观察检索效果。

Rerank 和 Embedding 检索的区别在于计算方式不同。Embedding 检索会先把文本预先编码成向量，查询时通过向量距离快速召回；Rerank 通常在候选集上逐条比较问题和文档，计算更精细，但成本更高。因此 Rerank 不适合直接对整个知识库排序，而适合放在候选召回之后，对几十条以内的候选结果做精排。

在当前 RAG 流程中，Rerank 的位置是：

1. 用户提出问题；
2. 系统进行向量检索或 Hybrid Search；
3. 得到 candidate_k 个候选 chunks；
4. Rerank 模型为每个候选 chunk 打相关性分；
5. 系统按 rerank_score 选择更适合进入上下文的片段；
6. 最终 chunks 被拼接成 context，交给大模型生成答案。

Rerank 可以改善几类常见问题。第一，向量检索召回了语义相近但不能直接回答问题的片段，Rerank 可以把更匹配问题意图的片段排到前面。第二，Hybrid Search 引入了关键词噪声，Rerank 可以降低只包含少量关键词但上下文不相关的结果。第三，当多个候选都来自同一主题时，Rerank 可以选择更完整、更直接回答问题的 chunk。

当前项目还使用 Dense-Preserving 策略：最终结果中会优先保留一部分基础向量检索结果，再用 Rerank 后的 Hybrid 候选补充剩余位置。这样做是为了避免关键词检索或重排序波动导致语义检索的稳定结果完全丢失。对于教育资料 RAG，这种策略可以在召回扩展和回答稳定性之间取得更好的平衡。

