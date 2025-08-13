import os
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# ────────────────────────── Config ──────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGES_DIR     = os.getenv("IMAGES_DIR", "data")
INDEX_PATH     = os.getenv("INDEX_PATH", "index.faiss")
META_PATH      = os.getenv("META_PATH", "meta.parquet")

if not OPENAI_API_KEY:
    st.stop()  # fail fast with a clear msg
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful sales rep at a clothing store. "
    "You will help customers find relevant clothing items based on their requirements. "
    "Be kind and respectful."
)

# ────────────────────────── Cache loaders ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_index_and_meta():
    if not os.path.exists(INDEX_PATH):
        st.error(f"Missing index at {INDEX_PATH}. Run build_index.py first.")
        st.stop()
    if not os.path.exists(META_PATH):
        st.error(f"Missing metadata at {META_PATH}. Run build_index.py first.")
        st.stop()
    index = faiss.read_index(INDEX_PATH)
    meta  = pd.read_parquet(META_PATH)
    return index, meta

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# ────────────────────────── Embedding helper ──────────────────────────
def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

# ────────────────────────── Retrieval ──────────────────────────
def search(index, meta_df, query: str, k: int = 5):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)
    rows = []
    for idx in I[0]:
        row = meta_df.iloc[int(idx)]
        rows.append(
            {
                "display_name": row["display name"],
                "description":  row["description"],
                "category":     row["category"],
                "image_path":   row["image_path"],
            }
        )
    return rows

# ────────────────────────── Streamlit UI ──────────────────────────
st.set_page_config(page_title="Clothes RAG Chat", page_icon="🧥", layout="wide")
st.title("🧵 Clothes RAG Chat")
st.caption("Type what you’re looking for and I’ll find the closest matches from your local dataset.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top‑K results", 1, 12, 5)
    show_images = st.checkbox("Show images", True)
    st.write(f"Index: `{INDEX_PATH}`")
    st.write(f"Meta: `{META_PATH}`")
    st.write(f"Images dir: `{IMAGES_DIR}`")

index, meta = load_index_and_meta()
llm = get_llm()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! What can I help you find today?")
    ]

# Render chat history
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(m.content)

# Input box
user_query = st.chat_input("e.g., I’m looking for a white formal t‑shirt")
if user_query:
    # append user message
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # retrieve
    results = search(index, meta, user_query, k=k)

    # image grid (optional)
    if show_images and results:
        cols = st.columns(min(5, len(results)))
        for i, item in enumerate(results):
            with cols[i % len(cols)]:
                path = item["image_path"]
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"Missing image: {path}")
                st.caption(f"**{item['display_name']}**\n\n*{item['category']}*\n\n{item['description'][:120]}...")

    # build context for LLM
    items_text = "\n".join(
        f"- {r['display_name']} ({r['category']}): {r['description']}"
        for r in results
    )

    # LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"User query: {user_query}\n\n"
                            f"Retrieved items (top {k}):\n{items_text}\n\n"
                            "Recommend the best 5 items in a helpful and polite manner. "
                            "If items are very similar, explain differences briefly (fit, material, use‑case)."
                        )
                    ),
                ]
            ).content
            st.markdown(reply)
            st.session_state.messages.append(AIMessage(content=reply))
