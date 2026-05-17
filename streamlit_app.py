\
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_core import RAGIndex, answer_question, build_index


st.set_page_config(page_title="AI 旅平險專員", layout="wide")

st.title("AI 旅平險專員")
st.caption("本地 Ollama llama3.1:8b + BGE-M3 中文 Embedding + PDF RAG")

with st.sidebar:
    st.header("設定")
    data_dir = Path(st.text_input("PDF 資料夾", "data"))
    index_dir = Path(st.text_input("索引資料夾", "index"))
    embedding = st.text_input("Embedding 模型", "BAAI/bge-m3")
    ollama_model = st.text_input("Ollama 模型", "llama3.1:8b")
    top_k = st.slider("Top-K", 3, 10, 6)
    rerank = st.checkbox("啟用 Reranker（較慢）", value=False)

    if st.button("建立 / 重建索引"):
        with st.spinner("正在解析 PDF 並建立向量索引..."):
            build_index(data_dir=data_dir, index_dir=index_dir, embedding_name=embedding)
        st.success("索引建立完成")

@st.cache_resource(show_spinner=False)
def load_index(index_path: str, emb: str):
    return RAGIndex(index_dir=Path(index_path), embedding_name=emb)

question = st.text_input("請輸入旅平險問題", "班機延誤5小時可以理賠嗎？")

if st.button("查詢"):
    with st.spinner("檢索條款並呼叫本地 LLM..."):
        idx = load_index(str(index_dir), embedding)
        result = answer_question(
            question=question,
            index=idx,
            model=ollama_model,
            top_k=top_k,
            rerank=rerank,
        )

    st.subheader("回答")
    st.write(result["answer"])

    st.subheader("檢索來源")
    st.dataframe(pd.DataFrame(result["citations"]), use_container_width=True)
