# Hybrid Search 混合检索

Hybrid Search 是把多种检索方式结合起来的一种 RAG 检索策略。单纯向量检索擅长语义匹配，可以找到表达不同但意思相近的内容；关键词检索擅长精确匹配，可以抓住模型名称、缩写、专有名词和具体术语。混合检索的目标是利用两者优势，提高候选 chunks 的召回质量。

当前项目中的 Hybrid Search 包含两个通道。第一个通道是 dense search，也就是向量检索。系统调用 ChromaDB，根据用户问题的 Embedding 向量查找相似 chunks，并把 Chroma 返回的 distance 转换为 dense_score。第二个通道是 sparse search，也就是轻量关键词检索。项目使用简单分词函数提取英文、数字和中文字符，计算 query 与 chunk 的关键词重叠数量，得到 sparse_score。

两个通道各自返回 candidate_k 条候选结果后，系统使用 RRF 排名融合。RRF 的核心思想不是直接比较原始分数，而是比较排名位置。排名越靠前，贡献越大。当前项目还使用加权 RRF，让 dense search 权重更高，sparse search 权重更低。这样可以保持语义检索为主，同时让关键词检索补充专有名词和精确表达。

Hybrid Search 的基本流程如下：

1. 将用户问题送入 dense search，得到向量检索候选；
2. 将用户问题送入 sparse search，得到关键词候选；
3. 对两个候选列表按 chunk_id 合并去重；
4. 使用 RRF 计算 hybrid_score；
5. 按 hybrid_score 排序，返回 Top-K 或 candidate_k 结果；
6. 如启用 Rerank，再把候选结果交给重排序模型。

混合检索适合处理两类问题。第一类是语义表达变化较大的问题，例如“向量库怎么帮 RAG 找资料”和“向量数据库在检索增强生成中的作用”。第二类是包含明确术语的问题，例如“BAAI/bge-m3”“Top-K”“RRF”“ChromaDB”。向量检索能覆盖语义相近表达，关键词检索能避免专有词被语义模型弱化。

Hybrid Search 也可能引入噪声。关键词重叠并不总是代表语义相关，尤其是中文按单字匹配时，常见字可能造成误召回。因此当前项目采用 dense 权重更高的策略，并在后续加入 Rerank。这样系统既能扩大候选范围，又能尽量避免无关关键词结果直接进入最终上下文。

