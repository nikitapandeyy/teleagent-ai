import streamlit as st
from agent import run_agent, FAQ, PLANS

st.set_page_config(page_title="TeleAgent AI", page_icon="📡", layout="centered")

st.title("📡 TeleAgent AI")
st.caption("An agentic RAG copilot for telecom customer support — billing, plans, WiFi, wallet & more.")

with st.sidebar:
    st.header("Setup")
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        secret_key = None
    if secret_key:
        api_key = secret_key
        st.success("Gemini API key loaded ✓")
    else:
        api_key = st.text_input("Google Gemini API Key", type="password", help="Free key: aistudio.google.com/app/apikey")
        st.caption("Uses Gemini's free tier — no billing required.")
    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        "1. **Classify** — Gemini tags your query's intent\n"
        "2. **Retrieve** — vector search (Gemini embeddings) over the FAQ base\n"
        "3. **Gate** — low confidence auto-raises a support ticket\n"
        "4. **Generate** — Gemini answers using only retrieved context\n"
    )
    st.markdown("---")
    st.subheader("Knowledge base")
    st.caption(f"{len(FAQ)} FAQ entries · {len(PLANS)} plans loaded")
    with st.expander("Try asking..."):
        st.markdown(
            "- Why was I charged extra on my last bill?\n"
            "- My home WiFi keeps disconnecting, help!\n"
            "- Recommend a plan for 3GB/day for 4 family members\n"
            "- How do I activate my OTT bundle?\n"
            "- Can I get satellite internet at my farmhouse? *(triggers escalation)*"
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 Agent trace"):
                for step in msg["trace"]:
                    st.markdown(f"- {step}")
        if msg.get("ticket"):
            t = msg["ticket"]
            st.warning(f"🎫 Ticket **{t['ticket_id']}** raised · {t['status']} · {t['created_at']}")

query = st.chat_input("Ask about your bill, plan, WiFi, wallet, or device...")

if query:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Routing through agent graph..."):
                try:
                    result = run_agent(query, api_key)
                    answer = result.get("answer", "Sorry, something went wrong.")
                    trace = result.get("agent_trace", [])
                    ticket = result.get("ticket")

                    st.markdown(answer)
                    with st.expander("🔍 Agent trace", expanded=True):
                        for step in trace:
                            st.markdown(f"- {step}")
                    if ticket:
                        st.warning(f"🎫 Ticket **{ticket['ticket_id']}** raised · {ticket['status']} · {ticket['created_at']}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "trace": trace, "ticket": ticket}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

if not st.session_state.messages:
    st.info("👋 Ask a telecom support question below to see the agent route through classification → retrieval → (escalation if needed) → answer generation.")
