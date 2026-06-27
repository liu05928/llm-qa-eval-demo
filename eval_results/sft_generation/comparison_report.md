# SFT Generation Evaluation Comparison

Question set: `data/sft_v2/quality_eval_questions.json` (50 held-out questions).

| run | avg_keyword_score | keyword_all_hit_rate | citation_complete_rate | structure_complete_rate | refusal_correct_rate | reference_mismatch_rate | avg_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_qwen25_3b | 0.595 | 0.34 | 0.76 | 0.58 | 0.3636 | 0.0 | 5.677 |
| v1_qlora | 0.5667 | 0.24 | 0.98 | 0.72 | 0.0 | 0.0 | 5.556 |
| v2_qlora | 0.76 | 0.44 | 1.0 | 1.0 | 0.0 | 0.0 | 7.5255 |

## Main Findings

- v2 is best on structure stability and citation completeness: both are 1.0.
- v2 improves average keyword score over v1/base, especially grounded QA and citation QA.
- Refusal remains the main failure: v2 and v1 score 0.0 refusal correctness, while base reaches 0.3636.
- No run showed reference mismatch under the source-path heuristic.

## Recommended Next Step

Create a v2.1 dataset patch focused on hard refusal samples. These should keep the same retrieved context format, but ask for unsupported facts such as real-time scores, future exam questions, private data, current prices, or personal medical/grade judgments. The target answer should explicitly say the reference material is insufficient before giving any study suggestion.
