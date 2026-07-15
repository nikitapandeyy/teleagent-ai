# TeleAgent AI — Agentic RAG Telecom Support Copilot

Built for **The Talent Hack** (Deutsche Telekom Digital Labs).

## What it is

A multi-agent, RAG-powered customer support copilot for telecom use cases — billing, plans,
WiFi troubleshooting, entertainment/OTT, digital wallet, and device financing — mirroring the
real product surfaces DTDL builds. The Gemini API key is baked in as a secret, so anyone opening
the deployed app can use it immediately with no setup on their end.

**Architecture (LangGraph):**

```
START -> classify_intent -> route -> retrieve_faq (vector RAG) -> confidence_gate -> escalate_to_human -> generate_answer -> END
                                                                                   -> generate_answer -> END
                                   -> recommend_plan (tool)                                             -> generate_answer -> END
```

- `classify_intent` — Gemini labels the query's intent (billing/plan/wifi/entertainment/wallet/device/general)
- `retrieve_faq` — **semantic vector search** over the FAQ base using Gemini embeddings
  (`text-embedding-004`), with an automatic TF-IDF fallback if the embeddings call ever fails
- `recommend_plan` — rule-based tool that matches stated data usage / number of lines to the
  cheapest fitting plan
- `confidence_gate` + `escalate_to_human` — if retrieval confidence falls below a threshold, the
  agent autonomously raises a support ticket instead of guessing
- `generate_answer` — Gemini synthesizes the final answer, grounded only in retrieved context

The UI shows a live **agent trace** for every response — the exact sequence of nodes the graph
visited — so it's obvious this is a real multi-step agent, not a single prompt.

## Tech stack

Python · Streamlit · LangChain · LangGraph · Google Gemini (chat + embeddings, free tier) · scikit-learn

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The key lives in `.streamlit/secrets.toml` (already filled in, and gitignored — it will never be
committed). If that file is missing, the app falls back to asking for a key in the sidebar.

## Deploy on Streamlit Community Cloud (free, ~5 minutes)

1. **Push this folder to a new GitHub repo.** Because `.streamlit/secrets.toml` is in
   `.gitignore`, your key stays off GitHub even if the repo is public:
   ```bash
   git init
   git add .
   git commit -m "TeleAgent AI - Talent Hack submission"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → select your repo → branch `main` → main file path `app.py`.
4. Before/after deploying, open **App settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
   This is the *cloud* copy of the secret — it's separate from your local `secrets.toml` and is
   what makes the key available to the live app without exposing it in the repo.
5. Click **Deploy**. You'll get a live URL like `https://<your-app>.streamlit.app`.
6. Open it and send a test query to confirm the sidebar shows "Gemini API key loaded ✓" and a
   response comes back with an agent trace.

> Security note: since this key was shared in plaintext during our chat, it's worth rotating it
> in Google AI Studio after the hackathon judging window, just as good hygiene.

## Submit on HackerEarth

On the challenge's "Submissions" tab, click **Start submission** and provide:
- Project name: `TeleAgent AI`
- One-line description: *Agentic RAG copilot for telecom support — routes queries through a
  LangGraph agent to vector-search retrieval, a plan recommender, or auto-escalation, then
  answers with Gemini.*
- GitHub repo link
- Live Streamlit app link
- (Optional) 1-2 min demo video/GIF showing: a billing question, a "recommend a plan for 3GB/day,
  4 lines" query, and an off-topic question to show the auto-escalation ticket firing

## Try these sample queries in the demo

- "Why was I charged extra on my last bill?"
- "My home WiFi keeps disconnecting, help!"
- "Recommend a plan for 3GB/day for 4 family members"
- "How do I activate my OTT bundle?"
- "Can I get satellite internet at my farmhouse?" — off-topic, triggers the escalation ticket

## Possible "if I have 10 more minutes" extensions

- Persist raised tickets to a Google Sheet or database instead of just showing them in-session
- Add conversation memory across turns using LangGraph's checkpointer
- Add a second retrieval index for the plan catalog so `recommend_plan` also cites plan FAQs
