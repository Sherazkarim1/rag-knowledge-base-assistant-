"""
RAG Knowledge Base Assistant — Streamlit UI
A clean, interactive web interface for uploading documents
and asking questions using the RAG pipeline.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Knowledge Base Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.markdown("**Upload your documents** and ask any question about them.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Indexing documents..."):
                files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
                r = requests.post(f"{API_URL}/upload", files=files)
                if r.status_code == 200:
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)!")
                else:
                    st.error(f"Error: {r.json().get('detail')}")

    st.divider()
    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        r = requests.delete(f"{API_URL}/reset")
        st.success("Knowledge base reset!")
        st.session_state.messages = []

# ── Chat Interface ────────────────────────────
st.title("💬 Ask Your Documents")
st.caption("Powered by GPT-4 + ChromaDB RAG Pipeline — Built by Sheraz Karim")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"• {s}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = requests.post(f"{API_URL}/ask", json={"question": prompt})
            if r.status_code == 200:
                data = r.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for s in sources:
                            st.write(f"• {s}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                err = r.json().get("detail", "Something went wrong.")
                st.error(err)
