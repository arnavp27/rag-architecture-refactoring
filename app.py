import streamlit as st
import sys
import logging
from pathlib import Path

# Add project root to path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from application.builders import RAGPipelineBuilder
from components.retrieval.strategies import VectorOnlyStrategy, KeywordOnlyStrategy, HybridStrategy

# Page Config
st.set_page_config(page_title="RAG Architect", page_icon="🤖", layout="wide")

# Custom CSS for a professional look
st.markdown("""
<style>
    .stChatMessage {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
    .stButton button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# --- 1. Initialize Pipeline (Cached) ---
@st.cache_resource
def get_pipeline():
    """
    Builds the pipeline once and caches it in memory.
    This demonstrates the Singleton/Builder pattern in action.
    """
    try:
        print("🔧 Building RAG Pipeline for UI...")
        pipeline = (RAGPipelineBuilder()
            .with_llm(primary="ollama", fallback="gemini")
            .with_embedder("sentence-transformers")
            .with_vector_store() # Uses .env settings
            .with_retrieval_strategy("hybrid")
            .with_caching(cache_size=100)
            .with_timing(threshold_ms=1000)
            .with_reranker("cross-encoder")
            .build())
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

pipeline = get_pipeline()

# --- 2. Sidebar: System Controls ---
with st.sidebar:
    st.header("⚙️ System Architecture")
    
    st.info("These controls demonstrate the **Strategy Pattern** allowing runtime behavior changes.")
    
    # Strategy Selector
    strategy_option = st.selectbox(
        "Retrieval Strategy",
        ["Hybrid (RRF)", "Vector Only (Semantic)", "Keyword Only (BM25)"],
        index=0
    )
    
    # Apply Strategy Change
    if pipeline:
        retriever = pipeline.get_retriever()
        vector_store = pipeline._vector_store
        
        if "Vector" in strategy_option:
            retriever.set_strategy(VectorOnlyStrategy(vector_store))
        elif "Keyword" in strategy_option:
            retriever.set_strategy(KeywordOnlyStrategy(vector_store))
        else:
            retriever.set_strategy(HybridStrategy(vector_store))
            
    st.divider()
    
    # Model Info
    st.subheader("🧩 Active Components")
    st.text("LLM: Ollama (Gemma-2b)")
    st.text("Fallback: Gemini Flash")
    st.text("Embedder: Qwen 0.6B")
    st.text("Reranker: MiniLM-L6")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 3. Chat Interface ---
st.title("🤖 Enterprise RAG System")
st.caption("Demonstrating 5-Layer Clean Architecture & Hybrid Search")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Retrieved Sources & Metrics"):
                st.markdown(message["sources"])

# User Input
if prompt := st.chat_input("Ask a question about politics..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not pipeline:
            st.error("Pipeline is not running.")
        else:
            with st.spinner("Thinking (Retrieving -> Reranking -> Generating)..."):
                try:
                    # Execute Query
                    response = pipeline.query(prompt)
                    
                    # Format Answer
                    st.markdown(response.answer)
                    
                    # Format Sources & Metrics
                    source_text = "### ⏱️ Performance Metrics\n"
                    source_text += f"- **Total Time:** {response.metrics.total_time_ms:.0f}ms\n"
                    source_text += f"- **Retrieval:** {response.metrics.retrieval_time_ms:.0f}ms\n"
                    source_text += f"- **Reranking:** {response.metrics.rerank_time_ms:.0f}ms\n"
                    source_text += f"- **Generation:** {response.metrics.generation_time_ms:.0f}ms\n\n"
                    
                    source_text += "### 🔍 Retrieved Context\n"
                    for i, doc in enumerate(response.sources, 1):
                        source_text += f"**[{i}] Score: {doc.score:.3f} | Rerank: {doc.rerank_score:.3f}**\n"
                        source_text += f"> {doc.content[:300]}...\n\n"
                    
                    # Save to history with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.answer,
                        "sources": source_text
                    })
                    
                    # Show sources for immediate view
                    with st.expander("📚 View Retrieved Sources & Metrics"):
                        st.markdown(source_text)
                        
                except Exception as e:
                    st.error(f"Error processing query: {e}")