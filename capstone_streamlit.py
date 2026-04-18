"""
capstone_streamlit.py — Clinical Policy & Factuality Agent · Streamlit UI
Run: streamlit run capstone_streamlit.py
"""

import uuid
import streamlit as st

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Policy Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local import (after page config) ─────────────────────────────────────────
from agent import get_app, get_embedder, get_collection, get_llm, DOCUMENTS

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCE INITIALIZATION  (@st.cache_resource — runs only once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def initialize_resources():
    """
    Heavy singletons: embedder, ChromaDB collection, LLM, compiled graph.
    Cached across reruns and users.
    """
    embedder   = get_embedder()      # SentenceTransformer('all-MiniLM-L6-v2')
    collection = get_collection()    # ChromaDB + seeded documents
    llm        = get_llm()           # ChatGroq llama-3.3-70b-versatile
    app        = get_app()           # Compiled LangGraph StateGraph
    return embedder, collection, llm, app


embedder, collection, llm, app = initialize_resources()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "thread_id"    not in st.session_state:
    st.session_state.thread_id    = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of result dicts for display
if "lc_messages"  not in st.session_state:
    st.session_state.lc_messages  = []   # LangChain HumanMessage/AIMessage objects

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def score_badge(score: float) -> str:
    if score >= 0.85:
        return f"🟢 {score:.2f}"
    elif score >= 0.70:
        return f"🟡 {score:.2f}"
    else:
        return f"🔴 {score:.2f}"


def render_metadata(entry: dict) -> None:
    cols = st.columns(4)
    with cols[0]:
        st.caption(f"**Faithfulness:** {score_badge(entry.get('eval_score', 0.0))}")
    with cols[1]:
        route = entry.get("route", "—")
        icon  = {"retrieve": "🔍", "tool": "🔧", "memory_only": "💬"}.get(route, "❓")
        st.caption(f"**Route:** {icon} {route}")
    with cols[2]:
        retries = entry.get("retry_count", 0)
        st.caption(f"**Retries:** {'🔁 ' * retries}{retries}")
    with cols[3]:
        sources = entry.get("sources", [])
        if sources:
            st.caption(f"**Sources:** {', '.join(sources[:2])}")
        elif entry.get("tool_output"):
            st.caption("**Tool:** Drug Interaction / Datetime")
        else:
            st.caption("**Sources:** —")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/hospital.png",
        width=64,
    )
    st.title("Clinical Policy\nAgent")
    st.caption("St. Mercy Hospital — Internal Guidelines Assistant")
    st.divider()

    st.subheader("📚 Knowledge Base")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['title']}")

    st.divider()
    st.subheader("🛠️ Tools Available")
    st.markdown(
        "- 🔬 **Drug Interaction Checker**\n"
        "- 🕐 **Current Datetime**"
    )

    st.divider()
    st.subheader("🔀 Routing Logic")
    st.markdown(
        "| Path | Trigger |\n"
        "|---|---|\n"
        "| `retrieve` | Clinical policy queries |\n"
        "| `tool` | Drug interactions / datetime |\n"
        "| `memory_only` | Casual / off-topic |"
    )

    st.divider()
    st.subheader("⚙️ Session")
    st.code(f"Thread: {st.session_state.thread_id[:12]}…", language=None)
    st.caption(f"Messages in memory: {len(st.session_state.lc_messages)}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 New Session"):
            st.session_state.thread_id    = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.lc_messages  = []
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.lc_messages  = []
            st.rerun()

    st.divider()
    # Export chat history — Windows-safe encoding
    if st.session_state.chat_history and st.button("💾 Export Chat"):
        lines = []
        for i, e in enumerate(st.session_state.chat_history, 1):
            lines.append(f"[{i}] Q: {e['query']}")
            lines.append(f"    A: {e['answer']}")
            lines.append(f"    Route: {e.get('route','—')} | Score: {e.get('eval_score',0):.2f}")
            lines.append("")
        export_path = "chat_export.txt"
        with open(export_path, "w", encoding="utf-8") as fh:  # Windows-compatible
            fh.write("\n".join(lines))
        st.success(f"Saved to {export_path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.title("🏥 Clinical Policy & Factuality Agent")
st.markdown(
    "> **Powered by** LangGraph · ChromaDB · SentenceTransformer · ChatGroq llama-3.3-70b  \n"
    "> Self-reflection evaluates every answer for faithfulness before delivery."
)

# ── Sample queries ────────────────────────────────────────────────────────────
with st.expander("💡 Sample Queries", expanded=False):
    samples = [
        "What are the five WHO moments for hand hygiene?",
        "Describe all steps of the Hour-1 Sepsis Bundle.",
        "What PPE is needed for a patient with active TB?",
        "What is the Morse Fall Scale threshold for high-risk patients?",
        "Check the drug interaction between warfarin and aspirin.",
        "What are restricted antibiotics that require prior authorization?",
        "How should a suspected transfusion reaction be managed?",
        "What identifiers are required for patient identification?",
        "What is today's date?",
        "What is the weather like today?",  # red-team: off-topic
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state["prefill"] = s
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant", avatar="🏥"):
        st.write(entry["answer"])
        render_metadata(entry)
        if entry.get("tool_output"):
            with st.expander("🔧 Tool Output"):
                st.code(entry["tool_output"], language=None)
        if entry.get("sources"):
            with st.expander("📄 Retrieved Sources"):
                for src in entry["sources"]:
                    st.markdown(f"- {src}")

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input(
    "Ask about a clinical policy, drug interaction, or type 'help'…",
    key="chat_input",
) or prefill

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant", avatar="🏥"):
        status_placeholder = st.empty()

        status_placeholder.status("🔀 Routing query…", expanded=False)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        initial_state = {
            "messages":    st.session_state.lc_messages,
            "query":       query,
            "retrieved":   "",
            "sources":     [],
            "answer":      "",
            "eval_score":  0.0,
            "retry_count": 0,
            "route":       "",
            "tool_output": "",
        }

        with st.spinner("Consulting clinical guidelines and evaluating faithfulness…"):
            result = app.invoke(initial_state, config=config)

        status_placeholder.empty()

        answer      = result.get("answer",      "Unable to generate a response.")
        sources     = result.get("sources",     [])
        eval_score  = result.get("eval_score",  0.0)
        retry_count = result.get("retry_count", 0)
        route       = result.get("route",       "")
        tool_output = result.get("tool_output", "")

        # Faithfulness warning banner
        if eval_score < 0.7:
            st.warning(
                f"⚠️ Low faithfulness score ({eval_score:.2f}) after {retry_count} retry attempt(s). "
                "Please verify this information with your supervisor or the relevant department.",
                icon="⚠️",
            )
        elif eval_score < 0.85:
            st.info(f"ℹ️ Moderate faithfulness ({eval_score:.2f}). Cross-check with source policy documents.")

        st.write(answer)
        render_metadata({
            "eval_score":  eval_score,
            "route":       route,
            "retry_count": retry_count,
            "sources":     sources,
            "tool_output": tool_output,
        })

        if tool_output:
            with st.expander("🔧 Tool Output"):
                st.code(tool_output, language=None)

        if sources:
            with st.expander("📄 Retrieved Policy Documents"):
                for src in sources:
                    st.markdown(f"- **{src}**")

        # Update session state
        st.session_state.lc_messages  = result.get("messages", st.session_state.lc_messages)
        st.session_state.chat_history.append({
            "query":       query,
            "answer":      answer,
            "sources":     sources,
            "eval_score":  eval_score,
            "retry_count": retry_count,
            "route":       route,
            "tool_output": tool_output,
        })
