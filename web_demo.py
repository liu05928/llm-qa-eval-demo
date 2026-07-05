import os
import json
import csv
from pathlib import Path

import streamlit as st

from agent_session_store import (
    get_or_create_session,
    get_session_snapshot,
    reset_session,
    save_session,
)
from langgraph_agent import run_langgraph_rag_agent
from rag_agent import run_rag_agent
from rag_pipeline import rag_answer
from config import CHAT_MODEL, EMBEDDING_MODEL, GENERATION_BACKEND, LOCAL_SFT_MODEL, RERANK_MODEL


st.set_page_config(
    page_title="教育领域大模型可信问答系统",
    page_icon="📚",
    layout="wide"
)


NO_KEYWORD_EVAL_FILE = Path("eval_results/no_keyword_dense_rerank_eval.csv")
BM25_EVAL_FILE = Path("eval_results/bm25_hybrid_rerank_eval.csv")
RETRIEVAL_LOG_FILE = Path("logs/retrieval_log.json")
AGENT_LOG_FILE = Path("logs/agent_trace_log.json")


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


def load_agent_logs():
    """读取 Agent 执行日志"""
    if not AGENT_LOG_FILE.exists():
        return []

    try:
        with AGENT_LOG_FILE.open("r", encoding="utf-8") as f:
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
        st.dataframe(rows, width="stretch")


st.title("📚 教育领域大模型可信问答系统")

st.markdown(
    f"""
本系统面向初中科学教材问答场景，主线是“模型微调 + 检索增强 + 可信回答”。当前本地版本支持 BM25 Hybrid、Rerank、Small-to-Big Long-text RAG、Workflow 编排、会话记忆、来源引用、检索日志和自动评测；云端 SFT 模型训练完成后可通过 OpenAI-compatible endpoint 接入。

**API baseline 模型：** `{CHAT_MODEL}`<br>
**默认生成后端：** `{GENERATION_BACKEND}`<br>
**SFT 模型占位：** `{LOCAL_SFT_MODEL}`<br>
**Embedding 模型：** `{EMBEDDING_MODEL}`<br>
**Rerank 模型：** `{RERANK_MODEL}`
"""
)

tab_agent, tab_chat, tab_eval, tab_log = st.tabs(
    ["Agent 问答演示", "RAG 问答演示", "评测结果展示", "检索日志查看"]
)


with tab_agent:
    st.header("Agent 问答演示")

    if "agent_session_id" not in st.session_state:
        session_id, _, _ = get_or_create_session()
        st.session_state.agent_session_id = session_id

    if "agent_dialogue" not in st.session_state:
        st.session_state.agent_dialogue = []

    col_left, col_right = st.columns([2, 1])

    with col_left:
        agent_question = st.text_area(
            "请输入 Agent 问题",
            value="什么是 RAG？",
            height=120,
            key="agent_question",
        )

    with col_right:
        agent_engine_label = st.selectbox(
            "Agent 编排引擎",
            options=["LangGraph Skills", "本地状态机"],
            index=0,
            help="LangGraph Skills 用于展示图编排工作流；本地状态机用于稳定 fallback 和回归对照。",
        )
        agent_engine = "langgraph" if agent_engine_label == "LangGraph Skills" else "local"

        enable_memory = st.checkbox(
            "启用会话记忆",
            value=True,
            key="enable_agent_memory",
        )

        agent_session_id = st.text_input(
            "Agent session_id",
            value=st.session_state.agent_session_id,
            key="agent_session_id_input",
            help="复用同一个 session_id 可以延续会话记忆。",
        ).strip()

        if agent_session_id != st.session_state.agent_session_id:
            st.session_state.agent_session_id = agent_session_id

        agent_top_k = st.slider(
            "Agent 最终文本块数量 top_k",
            min_value=1,
            max_value=5,
            value=3,
            key="agent_top_k",
        )

        agent_candidate_k = st.slider(
            "Agent 候选召回数量 candidate_k",
            min_value=3,
            max_value=15,
            value=10,
            key="agent_candidate_k",
        )

        agent_max_rewrites = st.slider(
            "最大 Query Rewrite 次数",
            min_value=0,
            max_value=2,
            value=1,
            key="agent_max_rewrites",
        )

        agent_use_rerank = st.checkbox(
            "Agent 启用模型 Rerank",
            value=True,
            key="agent_use_rerank",
        )

        agent_use_small_to_big = st.checkbox(
            "Agent 启用 Small-to-Big Long-text RAG",
            value=True,
            key="agent_use_small_to_big",
        )

        if st.button("清空 Agent 记忆", key="clear_agent_memory"):
            try:
                session_id, _, _ = get_or_create_session(
                    st.session_state.agent_session_id
                )
                st.session_state.agent_session_id = session_id
                reset_session(st.session_state.agent_session_id)
                st.session_state.agent_dialogue = []
                st.success("当前 session 记忆已清空。")
            except Exception as exc:
                st.error(f"清空记忆失败：{exc}")

    if st.button("启动 Agent 问答", type="primary", key="run_agent"):
        if not agent_question.strip():
            st.warning("请输入问题。")
        else:
            with st.spinner("Agent 正在判断任务、选择工具并生成回答..."):
                session_created = False
                memory = None

                if enable_memory:
                    session_id, memory, session_created = get_or_create_session(
                        st.session_state.agent_session_id
                    )
                    st.session_state.agent_session_id = session_id

                if agent_engine == "langgraph":
                    agent_result = run_langgraph_rag_agent(
                        question=agent_question.strip(),
                        top_k=agent_top_k,
                        candidate_k=agent_candidate_k,
                        max_rewrites=agent_max_rewrites,
                        use_rerank=agent_use_rerank,
                        context_mode="small_to_big" if agent_use_small_to_big else "small",
                        memory=memory,
                    )
                else:
                    agent_result = run_rag_agent(
                        question=agent_question.strip(),
                        top_k=agent_top_k,
                        candidate_k=agent_candidate_k,
                        max_rewrites=agent_max_rewrites,
                        use_rerank=agent_use_rerank,
                        context_mode="small_to_big" if agent_use_small_to_big else "small",
                        memory=memory,
                    )
                    agent_result["agent_engine"] = "local"

            if enable_memory:
                save_session(st.session_state.agent_session_id, memory)

            agent_result["session_id"] = st.session_state.agent_session_id
            agent_result["session_created"] = session_created

            st.session_state.agent_dialogue.append({
                "question": agent_question.strip(),
                "resolved_question": agent_result.get("resolved_question"),
                "answer": agent_result.get("answer", ""),
                "memory_used": agent_result.get("memory_used"),
                "query_type_label": agent_result.get("query_type_label"),
            })

            st.subheader("Agent 决策概览")

            col_a, col_b, col_c, col_d, col_e, col_f, col_g, col_h = st.columns(8)

            with col_a:
                st.metric("问题类型", agent_result.get("query_type_label", "未知"))

            with col_b:
                st.metric("检索策略", agent_result.get("retriever_mode", "未知"))

            with col_c:
                context_label = "充足" if agent_result.get("context_sufficient") else "不足"
                st.metric("上下文", context_label)

            with col_d:
                st.metric(
                    "覆盖度",
                    round(float(agent_result.get("context_coverage") or 0), 4),
                )

            with col_e:
                memory_label = "使用" if agent_result.get("memory_used") else "未使用"
                st.metric("记忆", memory_label)

            with col_f:
                st.metric("session", "新建" if agent_result.get("session_created") else "复用")

            with col_g:
                st.metric("上下文模式", agent_result.get("context_mode", "small"))

            with col_h:
                st.metric("引擎", agent_result.get("agent_engine", agent_engine))

            st.caption(f"session_id: `{agent_result.get('session_id')}`")
            st.caption(agent_result.get("query_reason", ""))
            st.caption(agent_result.get("memory_reason", ""))
            st.caption(agent_result.get("context_reason", ""))

            if agent_result.get("resolved_question") != agent_result.get("question"):
                st.markdown(
                    f"**记忆补全问题：** `{agent_result.get('resolved_question')}`"
                )

            rewritten_queries = agent_result.get("rewritten_queries", [])

            if rewritten_queries:
                with st.expander("查看 Query Rewrite"):
                    for query in rewritten_queries:
                        st.markdown(f"- `{query}`")

            st.subheader("Agent 回答")
            st.write(agent_result.get("answer", ""))

            st.subheader("引用来源")
            agent_sources = agent_result.get("sources", [])

            if agent_sources:
                for source in agent_sources:
                    st.markdown(
                        f"- `{source.get('source')}` / `{source.get('chunk_id')}`"
                    )
            else:
                st.info("Agent 判断当前回答不应引用低相关来源。")

            st.subheader("Agent 工具调用轨迹")
            agent_trace = agent_result.get("agent_trace", [])

            if agent_trace:
                st.dataframe(agent_trace, width="stretch")

                graph_trace = agent_result.get("graph_trace") or []

                if graph_trace:
                    st.caption("Graph 节点轨迹：" + " -> ".join(graph_trace))

                skill_trace = agent_result.get("skill_trace") or []

                if skill_trace:
                    with st.expander("查看 Skills 结构化轨迹"):
                        st.dataframe(skill_trace, width="stretch")

                with st.expander("查看工具调用详情"):
                    for step in agent_trace:
                        st.markdown(
                            f"**{step.get('node')}** | `{step.get('tool_name')}` | "
                            f"{step.get('status')} | {step.get('elapsed_ms')} ms"
                        )
                        st.write(step.get("output_summary", ""))
                        metadata = step.get("metadata") or {}

                        if metadata:
                            st.json(metadata)
            else:
                st.info("暂无 Agent 轨迹。")

            st.subheader("Agent 检索片段")

            agent_chunks = agent_result.get("retrieved_chunks", [])

            for i, chunk in enumerate(agent_chunks, start=1):
                with st.expander(
                    f"Top {i} | {chunk.get('source')} | {chunk.get('chunk_id')}"
                ):
                    col_1, col_2, col_3, col_4, col_5 = st.columns(5)

                    with col_1:
                        st.metric(
                            "dense_score",
                            round(float(chunk.get("dense_score") or 0), 4),
                        )

                    with col_2:
                        st.metric(
                            "bm25_score",
                            round(float(chunk.get("bm25_score") or 0), 4),
                        )

                    with col_3:
                        st.metric(
                            "hybrid_score",
                            round(float(chunk.get("hybrid_score") or 0), 4),
                        )

                    with col_4:
                        rerank_score = chunk.get("rerank_score")

                        if rerank_score is None:
                            st.metric("rerank_score", "None")
                        else:
                            st.metric("rerank_score", round(float(rerank_score), 4))

                    with col_5:
                        combined_score = chunk.get("rerank_combined_score")

                        if combined_score is None:
                            st.metric("combined", "None")
                        else:
                            st.metric("combined", round(float(combined_score), 4))

                    st.caption(
                        f"chunk_type={chunk.get('chunk_type')} | "
                        f"context_mode={chunk.get('context_mode')} | "
                        f"trigger_count={chunk.get('trigger_count', 0)}"
                    )

                    trigger_chunk_ids = chunk.get("trigger_chunk_ids") or []

                    if trigger_chunk_ids:
                        st.markdown("**触发父段落的小 chunk：**")
                        st.code("\n".join(trigger_chunk_ids), language="text")

                    st.markdown("**片段内容：**")
                    st.write(chunk.get("content", ""))

    try:
        memory_snapshot = get_session_snapshot(
            st.session_state.agent_session_id
        ).get("memory", {})
    except Exception:
        memory_snapshot = {}

    with st.expander("查看当前 Agent 记忆"):
        st.json({
            "session_id": st.session_state.get("agent_session_id"),
            "current_topic": memory_snapshot.get("current_topic"),
            "current_topics": memory_snapshot.get("current_topics"),
            "recent_turn_count": len(memory_snapshot.get("recent_turns", [])),
            "recent_turns": memory_snapshot.get("recent_turns", [])[-3:],
        })

    if st.session_state.get("agent_dialogue"):
        with st.expander("查看本轮页面对话历史"):
            for item in st.session_state.agent_dialogue[-5:]:
                st.markdown(
                    f"**Q:** {item.get('question')}  \n"
                    f"**Resolved:** `{item.get('resolved_question')}`  \n"
                    f"**Memory:** `{item.get('memory_used')}` | "
                    f"**Type:** `{item.get('query_type_label')}`"
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
            options=["vector", "dense_rerank", "bm25_hybrid", "contextual_hybrid"],
            index=2,
            help="vector 表示基础向量检索；dense_rerank 表示无关键词召回对照；bm25_hybrid 表示向量召回 + BM25 + Rerank；contextual_hybrid 会在 BM25 侧加入来源和章节上下文。"
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
            "启用模型 Rerank",
            value=True
        )

        use_small_to_big = st.checkbox(
            "启用 Small-to-Big Long-text RAG",
            value=True,
            help="小 chunk 用于召回，父级大段落用于回答。"
        )

        guard_mode = st.selectbox(
            "上下文 Guard",
            options=["v2", "v1"],
            index=0,
            help="v2 会输出 support_level、evidence_score 和 claim_verification；v1 为旧覆盖率判断。",
        )

        generation_backend = st.selectbox(
            "生成模型后端",
            options=["默认配置", "mock", "api", "local_sft"],
            index=0,
            help="local_sft 用于后续接入云端微调后的 Qwen LoRA/QLoRA 服务。",
        )

        selected_generation_backend = (
            None if generation_backend == "默认配置" else generation_backend
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
                    context_mode="small_to_big" if use_small_to_big else "small",
                    generation_backend=selected_generation_backend,
                    guard_mode=guard_mode,
                )

            st.subheader("模型回答")
            st.write(result["answer"])
            st.caption(
                f"生成后端：`{result.get('generation_backend')}` | "
                f"生成模型：`{result.get('generator_model')}`"
            )

            guard_cols = st.columns(4)
            guard_cols[0].metric("support_level", result.get("support_level", ""))
            guard_cols[1].metric(
                "evidence_score",
                round(float(result.get("evidence_score") or 0), 4),
            )
            guard_cols[2].metric(
                "coverage",
                round(float(result.get("context_coverage") or 0), 4),
            )
            guard_cols[3].metric(
                "claim_status",
                (result.get("claim_verification") or {}).get("status", ""),
            )

            with st.expander("查看 Guard 细节"):
                st.json({
                    "context_sufficient": result.get("context_sufficient"),
                    "context_reason": result.get("context_reason"),
                    "guard_mode": result.get("guard_mode"),
                    "guard_details": result.get("guard_details"),
                    "claim_verification": result.get("claim_verification"),
                })

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
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)

                    with col_a:
                        st.metric(
                            "dense_score",
                            round(float(chunk.get("dense_score") or 0), 4)
                        )

                    with col_b:
                        st.metric(
                            "bm25_score",
                            round(float(chunk.get("bm25_score") or 0), 4)
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

                    with col_e:
                        combined_score = chunk.get("rerank_combined_score")
                        if combined_score is None:
                            st.metric("combined", "None")
                        else:
                            st.metric("combined", round(float(combined_score), 4))

                    st.caption(
                        f"chunk_type={chunk.get('chunk_type')} | "
                        f"context_mode={chunk.get('context_mode')} | "
                        f"trigger_count={chunk.get('trigger_count', 0)}"
                    )

                    trigger_chunk_ids = chunk.get("trigger_chunk_ids") or []

                    if trigger_chunk_ids:
                        st.markdown("**触发父段落的小 chunk：**")
                        st.code("\n".join(trigger_chunk_ids), language="text")

                    rerank_model = chunk.get("rerank_model")

                    if rerank_model:
                        st.markdown(f"**Rerank 模型：** `{rerank_model}`")
                    else:
                        st.markdown("**Rerank 模型：** `未使用 / 无模型打分`")

                    st.markdown("**片段内容：**")
                    st.write(chunk.get("content", ""))


with tab_eval:
    st.header("评测结果展示")

    no_keyword_rows = load_csv_rows(NO_KEYWORD_EVAL_FILE)
    bm25_rows = load_csv_rows(BM25_EVAL_FILE)

    show_eval_summary("No Keyword：Dense Rerank", no_keyword_rows)

    st.divider()

    show_eval_summary("BM25 Hybrid + 模型 Rerank", bm25_rows)


with tab_log:
    st.header("日志查看")

    agent_logs = load_agent_logs()
    logs = load_retrieval_logs()

    if agent_logs:
        latest_agent_log = agent_logs[-1]

        st.subheader("最近一次 Agent 执行记录")

        st.json({
            "time": latest_agent_log.get("agent_log_time"),
            "question": latest_agent_log.get("question"),
            "query_type": latest_agent_log.get("query_type_label"),
            "retriever_mode": latest_agent_log.get("retriever_mode"),
            "context_sufficient": latest_agent_log.get("context_sufficient"),
            "context_coverage": latest_agent_log.get("context_coverage"),
            "rewritten_queries": latest_agent_log.get("rewritten_queries"),
        })

        with st.expander("查看 Agent 工具调用轨迹"):
            st.json(latest_agent_log.get("agent_trace", []))

    if not logs:
        st.info("暂未生成 RAG 检索日志。请先在问答页面运行一次问题。")
    else:
        latest_log = logs[-1]

        st.subheader("最近一次 RAG 检索记录")

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
