# 科学教材 Small-to-Big 对比实验报告

## 实验设置

- 测试集：`data/science_rag_test_questions.json`，共 30 条初中科学教材问题。
- 召回策略：`retriever_mode=bm25_hybrid`，`top_k=3`，`candidate_k=10`，`use_rerank=True`。
- 对照组：`context_mode=small`，小 chunk 直接用于回答。
- 实验组：`context_mode=small_to_big`，小 chunk 召回后扩展到父级大 chunk 用于回答。

关键词评测同时保留两种口径：严格关键词命中只做字面匹配；同义关键词命中支持少量教材领域别名，例如 `固态≈固体形态`、`气态≈气体形态`。
资料缺失题不参与来源命中率、关键词命中率和引用完整率统计，单独统计资料缺失拒答率。

## 指标对比

| 指标 | small | small_to_big | 差值 |
| --- | ---: | ---: | ---: |
| 来源命中率 | 100.00% | 100.00% | 0.00% |
| 严格关键词命中率 | 100.00% | 100.00% | 0.00% |
| 同义关键词命中率 | 100.00% | 100.00% | 0.00% |
| 引用完整率 | 100.00% | 100.00% | 0.00% |
| 资料缺失拒答率 | 100.00% | 100.00% | 0.00% |
| 平均回答长度 | 434.1 | 438.6 | 4.5 |
| 平均上下文字符数 | 1237.4 | 1676.1 | 438.7 |
| 平均小 chunk 数 | 3.0 | 3.0 | 0.0 |
| 平均父级大 chunk 数 | 0.0 | 2.1 | 2.1 |
| 平均触发小 chunk 数 | 0.0 | 3.0 | 3.0 |

## 结论

- Small-to-Big 不改变小 chunk 的召回入口，因此来源命中率主要反映检索质量。
- Small-to-Big 显著增加回答阶段可用上下文长度，适合教材、政策、公文制度等长文本资料。
- 同义关键词命中用于降低表达变体造成的假阴性，严格关键词命中仍保留用于排查回答措辞变化。
- 资料缺失拒答率用于验证系统边界意识，避免把无关教材片段强行拼成答案。
- 页面和日志会保留 `trigger_chunk_ids`，可追溯父级大段落由哪些小 chunk 命中触发。

## 结果文件

- `eval_results/science_small_context_eval.csv`
- `eval_results/science_small_to_big_eval.csv`
- `eval_results/science_small_to_big_summary.json`
- `eval_results/science_failure_cases.md`
