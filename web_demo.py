import os
import json
import csv
from pathlib import Path

import streamlit as st

from rag_pipeline import rag_answer


st.set_page_config(
    page_title="教育资料 RAG 知识库问答系统",
    page_icon="📚",
    layout="wide"
)


BASELINE_EVAL_FILE = Path("eval_results/baseline_eval.csv")
HYBRID_EVAL_FILE = Path("eval_results/hybrid_rerank_eval.csv")
RETRIEVAL_LOG_FILE = Path("logs/retrieval_log.json")


def load_csv_rows(file_path: Path):
    """读取 CSV 文件，返回 list[dict]"""
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def calc_rate(rows, field_name):
    """计算 CSV 中某个布尔字段的 True 比例"""
    if not rows:
        return 0

    valid_values = []

    for row in rows:
        value = str(row.get(field_name, "")).lower()

        if value in ["true", "false"]:
            valid_values.append(value == "true")

    if not valid_values:
        return 0

    return sum(valid_values) / len(valid_values)


def load_retrieval_logs():
    """读取检索日志"""
    if not RETRIEVAL_LOG_FILE.exists():
        return []

    try:
        with RETRIEVAL_LOG_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def show_eval_summary(title, rows):
    """展示评测结果汇总"""
    st.subheader(title)

    if not rows:
        st.info("暂未生成该评测结果。")
        return

    col1, col2, col3, col4 = st.columns(4)

    source_hit_rate = calc_rate(rows, "source_hit")
    keyword_hit_rate = calc_rate(rows, "keyword_hit")
    has_citation_rate = calc_rate(rows, "has_citation")
    no_context_reject_rate = calc_rate(rows, "no_context_reject")

    with col1:
        st.metric("来源命中率", f"{source_hit_rate:.2%}")

    with col2:
        st.metric("关键词命中率", f"{keyword_hit_rate:.2%}")

    with col3:
        st.metric("引用完整率", f"{has_citation_rate:.2%}")

    with col4:
        st.metric("无资料拒答率", f"{no_context_reject_rate:.2%}")

    with st.expander("查看详细评测结果"):
        st.dataframe(rows, use_container_width=True)


st.title("📚 教育资料 RAG 知识库问答与评测优化系统")

st.markdown(
    """
本系统支持基础向量检索、Hybrid Search、Rerank、来源引用、检索日志记录和自动评测展示。
"""
)

tab_chat, tab_eval, tab_log = st.tabs(
    ["RAG 问答演示", "评测结果展示", "检索日志查看"]
)


with tab_chat:
    st.header("RAG 问答演示")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        question = st.text_area(
            "请输入问题",
            value="什么是 RAG？",
            height=120
        )

    with col_right:
        retriever_mode = st.selectbox(
            "检索模式",
            options=["vector", "hybrid"],
            index=1,
            help="vector 表示基础向量检索；hybrid 表示 Hybrid Search + Rerank。"
        )

        top_k = st.slider(
            "最终使用的文本块数量 top_k",
            min_value=1,
            max_value=5,
            value=3
        )

        candidate_k = st.slider(
            "候选召回数量 candidate_k",
            min_value=3,
            max_value=15,
            value=10
        )

        use_rerank = st.checkbox(
            "启用 Rerank",
            value=True
        )

    if st.button("开始问答", type="primary"):
        if not question.strip():
            st.warning("请输入问题。")
        else:
            with st.spinner("正在检索知识库并生成回答..."):
                result = rag_answer(
                    question=question.strip(),
                    top_k=top_k,
                    retriever_mode=retriever_mode,
                    candidate_k=candidate_k,
                    use_rerank=use_rerank,
                )

            st.subheader("模型回答")
            st.write(result["answer"])

            st.subheader("引用来源")
            sources = result.get("sources", [])

            if sources:
                for source in sources:
                    st.markdown(
                        f"- `{source.get('source')}` / `{source.get('chunk_id')}`"
                    )
            else:
                st.info("当前回答没有返回引用来源。")

            st.subheader("检索片段")

            retrieved_chunks = result.get("retrieved_chunks", [])

            for i, chunk in enumerate(retrieved_chunks, start=1):
                with st.expander(
                    f"Top {i} | {chunk.get('source')} | {chunk.get('chunk_id')}"
                ):
                    col_a, col_b, col_c, col_d = st.columns(4)

                    with col_a:
                        st.metric(
                            "dense_score",
                            round(float(chunk.get("dense_score") or 0), 4)
                        )

                    with col_b:
                        st.metric(
                            "sparse_score",
                            round(float(chunk.get("sparse_score") or 0), 4)
                        )

                    with col_c:
                        st.metric(
                            "hybrid_score",
                            round(float(chunk.get("hybrid_score") or 0), 4)
                        )

                    with col_d:
                        rerank_score = chunk.get("rerank_score")
                        if rerank_score is None:
                            st.metric("rerank_score", "None")
                        else:
                            st.metric("rerank_score", round(float(rerank_score), 4))

                    st.markdown("**片段内容：**")
                    st.write(chunk.get("content", ""))


with tab_eval:
    st.header("评测结果展示")

    baseline_rows = load_csv_rows(BASELINE_EVAL_FILE)
    hybrid_rows = load_csv_rows(HYBRID_EVAL_FILE)

    show_eval_summary("Baseline：基础向量检索", baseline_rows)

    st.divider()

    show_eval_summary("Optimized：Hybrid Search + Rerank", hybrid_rows)


with tab_log:
    st.header("检索日志查看")

    logs = load_retrieval_logs()

    if not logs:
        st.info("暂未生成检索日志。请先在问答页面运行一次问题。")
    else:
        latest_log = logs[-1]

        st.subheader("最近一次检索记录")

        st.json({
            "time": latest_log.get("time"),
            "question": latest_log.get("question"),
            "retriever_mode": latest_log.get("retriever_mode"),
            "candidate_k": latest_log.get("candidate_k"),
            "final_top_k": latest_log.get("final_top_k"),
            "use_rerank": latest_log.get("use_rerank"),
            "answer_length": latest_log.get("answer_length"),
        })

        st.subheader("最终上下文")
        st.json(latest_log.get("final_context", []))

        with st.expander("查看候选召回片段"):
            st.json(latest_log.get("retrieved_candidates", []))