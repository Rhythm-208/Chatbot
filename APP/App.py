import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingestion import ingest_pdf
from Chatbot import ChatEngine
load_dotenv()

st.set_page_config(page_title = "Multimodal PDF Chat",layout = "wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
SUMMARY_PERSIST_DIR = os.path.join(SCRIPT_DIR, "summary_chroma_db")

# ---------- Cached resources (created once per server process) ----------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_stores():
    embeddings = get_embeddings()
    chunks_store = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    summary_store = Chroma(
        collection_name="pdf_summaries",
        embedding_function=embeddings,
        persist_directory=SUMMARY_PERSIST_DIR,
    )
    return chunks_store, summary_store

chunks_store, summary_store = get_stores()

def get_all_sources() -> list[str]:
    """
    Pulls every distinct `source` filename already in the chunks store,
    so files uploaded in past sessions still show up as selectable.
    """
    try:
        raw = chunks_store.get(include = ["metadatas"])
        sources = {m.get("source") for m in raw.get("metadatas", []) if m.get("source")}
        return sorted(sources)

    except Exception:
        return []


#Sesssiom State

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine(chunks_store, summary_store)

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = get_all_sources()

if "messages" not in st.session_state:
    st.session_state.messages = []




with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or mode PDFs",type = ["pdf"] , accept_multiple_files = True
    )
    if uploaded_files:
        already_ingested = set(get_all_sources())
        for uf in uploaded_files:
            if uf.name in st.session_state.ingested_files or uf.name in already_ingested:
                st.sidebar.caption(f"⏭️ Skipping {uf.name} — already ingested previously.")
                if uf.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uf.name)
                continue

            with tempfile.NamedTemporaryFile(delete=False , suffix = ".pdf") as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name

            status = st.status(f"Ingesting {uf.name}...", expanded=True)

            def progress_cb(current,total,message):
                status.update(label = f"{uf.name}: {message}  ({current}/{total})")

            try:
                num_chunks , source_name = ingest_pdf(
                    tmp_path,chunks_store,summary_store, source_name = uf.name ,progress_callback=progress_cb
                )
                st.session_state.ingested_files.append(source_name)
                status.update(label=f"✅ {uf.name}: {num_chunks} chunks indexed", state="complete")

            except Exception as e:
                status.update(label = f"❌ Failed to ingest {uf.name}: {e}",state="error")

            finally:
                os.unlink(tmp_path)

    st.divider()
    st.subheader("Files to Search")
    all_sources = sorted(set(get_all_sources()) | set(st.session_state.ingested_files))

    if not all_sources:
        st.caption("No files uploaded")
        selected_sources = []

    else:
        select_all = st.checkbox("All files",value=True)
        if select_all:
            selected_sources = all_sources
            st.multiselect("Files in scope",all_sources,default = all_sources , disabled = True)

        else:
            selected_sources = st.multiselect("Files in scope",all_sources)

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_engine.chat_history = []
        st.rerun()


st.title("Multimodal PDF Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    prefix = f"[{s['citation_id']}] " if "citation_id" in s else ""
                    if "page" in s and s.get("page") is not None:
                        st.caption(f"{prefix}{s['source']} — page {s['page']}")
                    else:
                        st.caption(f"{prefix}{s['source']}")
query = st.chat_input("Ask a question about your uploaded PDFs...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chat_engine.ask(query, active_sources=selected_sources)
        st.markdown(result["answer"])
        st.caption(f"Mode: {result['mode']}")
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    prefix = f"[{s['citation_id']}] " if "citation_id" in s else ""
                    if "page" in s and s.get("page") is not None:
                        st.caption(f"{prefix}{s['source']} — page {s['page']}")
                    else:
                        st.caption(f"{prefix}{s['source']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })















