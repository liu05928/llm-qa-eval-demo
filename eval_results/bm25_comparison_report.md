# BM25 对比实验报告

## 实验设置

- No Keyword：`retriever_mode=dense_rerank`，向量召回后直接 Rerank。
- BM25 Hybrid：`retriever_mode=bm25_hybrid`，向量召回 + BM25 召回 + RRF 融合后 Rerank。
- 两组均使用 `top_k=3`、`candidate_k=10`、`use_rerank=True`。

## 指标对比

| 指标 | No Keyword | BM25 Hybrid | BM25 - No Keyword |
| --- | ---: | ---: | ---: |
| 来源命中率 | 96.00% | 96.00% | 0.00% |
| 关键词命中率 | 63.33% | 61.67% | -1.67% |
| 引用完整率 | 100.00% | 100.00% | 0.00% |
| 无资料拒答率 | 70.00% | 90.00% | 20.00% |
| 平均回答长度 | 581.2 | 576.6 | -4.6 |
| 平均检索片段数 | 3.0 | 3.0 | 0.0 |

## 结果文件

- `eval_results/no_keyword_dense_rerank_eval.csv`
- `eval_results/bm25_hybrid_rerank_eval.csv`
- `eval_results/bm25_comparison_summary.json`
