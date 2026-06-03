import streamlit as st

from rag_pipeline import rag_answer


st.set_page_config(
    page_title="教育资料 RAG 知识库问答系统",
    page_icon="📚",
    layout="wide",
)


st.title("📚 教育资料 RAG 知识库问答系统")

st.markdown(
    """
本系统基于本地教育资料构建 RAG 问答流程，支持：

- 本地文档读取；
- 文本切分；
- 向量检索；
- 基于检索内容生成回答；
- 返回引用来源；
- 展示检索片段。
"""
)


st.divider()


question = st.text_input(
    "请输入你的问题：",
    placeholder="例如：什么是 RAG？",
)

top_k = st.slider(
    "选择检索返回的文本片段数量 top_k：",
    min_value=1,
    max_value=5,
    value=3,
)


if st.button("开始问答"):
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        with st.spinner("正在检索知识库并生成回答，请稍等..."):
            result = rag_answer(
                question=question,
                top_k=top_k,
            )

        st.subheader("一、模型回答")
        st.write(result["answer"])

        st.subheader("二、引用来源")

        sources = result.get("sources", [])

        if sources:
            for index, source in enumerate(sources, start=1):
                st.markdown(
                    f"{index}. `{source['source']}` / `{source['chunk_id']}`"
                )
        else:
            st.info("当前没有返回引用来源。")

        st.subheader("三、检索到的知识片段")

        retrieved_chunks = result.get("retrieved_chunks", [])

        if retrieved_chunks:
            for index, chunk in enumerate(retrieved_chunks, start=1):
                with st.expander(
                    f"Top {index} | {chunk['source']} | {chunk['chunk_id']}"
                ):
                    st.markdown(f"**来源文件：** `{chunk['source']}`")
                    st.markdown(f"**片段 ID：** `{chunk['chunk_id']}`")
                    st.markdown(f"**距离 distance：** `{chunk.get('distance')}`")
                    st.markdown("**文本内容：**")
                    st.write(chunk["content"])
        else:
            st.info("当前没有检索到相关片段。")


st.divider()

st.markdown(
    """
### 当前项目状态说明

当前项目默认使用 Mock 模式跑通完整 RAG 工程链路。  
后续可接入 DeepSeek、通义千问或 OpenAI 等真实大模型 API，以生成更自然的回答。
"""
)