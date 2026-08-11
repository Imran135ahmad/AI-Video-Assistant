import streamlit as st
from dotenv import load_dotenv
from main import run_pipeline
from core.rag_engine import ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI Video Assistant")
st.caption("Transcribe, summarize, and chat with your video or audio content.")

# ── Sidebar: input controls ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Paste a YouTube URL or a local file path.\n"
        "2. Choose the spoken language.\n"
        "3. Click **Run Pipeline**.\n"
        "4. Explore the results and chat with your content."
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.warning("Please enter a YouTube URL or file path before running.")
    else:
        st.session_state.result = None
        st.session_state.messages = []

        with st.spinner("Processing… this may take a few minutes ⏳"):
            try:
                result = run_pipeline(source.strip(), language=language)
                st.session_state.result = result
                st.success("Pipeline complete!")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result

if result:
    st.header(f"📌 {result['title']}")
    st.divider()

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(result["summary"])

    with tab_actions:
        st.markdown(result["action_item"])

    with tab_decisions:
        st.markdown(result["key_decision"])

    with tab_questions:
        st.markdown(result["open_question"])

    with tab_transcript:
        st.text_area(
            "Full transcript",
            value=result["transcript"],
            height=400,
            label_visibility="collapsed",
        )
        st.download_button(
            "⬇️ Download transcript",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.subheader("Chat with your content")

        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about the video…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], prompt)
                    except Exception as exc:
                        answer = f"Error: {exc}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Enter a source and click **Run Pipeline** in the sidebar to get started.")
