"""
Phase 5: Streamlit interface for the debunker.

Type a claim, get a verdict backed by retrieved evidence -- each piece shown
with its source, relevance, and whether it supports or refutes the claim.

Run locally:
    export GOOGLE_FACTCHECK_KEY=...   TAVILY_KEY=...
    streamlit run app.py
  (or put both keys in a .env file -- loaded automatically below)

Deploy: Hugging Face Spaces (Streamlit SDK). Set the two keys as Space secrets.
"""

from __future__ import annotations

# Load .env BEFORE importing pipeline modules -- they read API keys at import
# time, so the env must be populated first. No-op if python-dotenv isn't
# installed or no .env exists (e.g. on Spaces, where secrets are already set).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from stance import Verifier
from pipeline import debunk

st.set_page_config(page_title="Claim Debunker", page_icon="🔍",
                   layout="centered")

# Verdict -> (emoji, colour) for the headline card.
VERDICT_STYLE = {
    "SUPPORTED": ("✅", "#1a7f37"),
    "REFUTED": ("❌", "#cf222e"),
    "NOT_ENOUGH_INFO": ("❓", "#9a6700"),
    "CONFLICTING": ("⚖️", "#8250df"),
}
EVIDENCE_MARK = {
    "SUPPORTS": "🟢 supports",
    "REFUTES": "🔴 refutes",
    "NEUTRAL": "⚪ doesn't resolve",
    "IRRELEVANT": "⚫ off-topic (filtered)",
}


@st.cache_resource(show_spinner="Loading models (first run only)…")
def get_verifier() -> Verifier:
    """Load models once and reuse across reruns -- critical on a shared host."""
    return Verifier()


def verdict_card(result) -> None:
    emoji, colour = VERDICT_STYLE.get(result.verdict, ("•", "#57606a"))
    st.markdown(
        f"<div style='padding:1.2rem 1.4rem;border-radius:12px;"
        f"border:1px solid {colour}33;background:{colour}11;'>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{colour};'>"
        f"{emoji}&nbsp;{result.headline}</div>"
        f"<div style='color:#57606a;margin-top:.3rem;'>"
        f"Verdict: <code>{result.verdict}</code>"
        + (f" · confidence {result.confidence:.0%}"
           f" · {result.n_voted} of {len(result.evidence)} sources voted"
           if result.evidence else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def evidence_row(e) -> None:
    dim = e.verdict in ("NEUTRAL", "IRRELEVANT")
    title = e.title or e.url or "source"
    with st.container(border=True):
        top = f"**{EVIDENCE_MARK.get(e.verdict, e.verdict)}**"
        top += f" &nbsp;·&nbsp; <span style='color:#57606a'>{e.source}</span>"
        st.markdown(top, unsafe_allow_html=True)
        st.markdown(("*" + e.text[:280] + "*") if dim else e.text[:280])
        meta = f"relevance {e.relevance:+.1f}"
        if e.stance:
            meta += f" · stance: {e.stance} ({e.stance_conf:.0%})"
        st.caption(meta)
        if e.url:
            st.markdown(f"[source ↗]({e.url})")


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

st.title("🔍 Claim Debunker")
st.markdown(
    "Enter a factual claim. The system retrieves evidence from fact-checkers "
    "and the web, filters it for relevance, and judges whether the evidence "
    "**supports** or **refutes** the claim — showing its work."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Retrieve** — fact-check API + web search\n"
        "2. **Gate** — drop evidence not about the claim\n"
        "3. **Stance** — NLI judges support / refute / neutral\n"
        "4. **Aggregate** — verdict from corroborating evidence"
    )
    st.markdown("---")
    st.caption(
        "A verdict of *not enough evidence* is a real answer: the system "
        "abstains rather than guess. It reports what the evidence shows, not "
        "ground truth."
    )
    max_web = st.slider("Web results to retrieve", 3, 10, 5)

examples = [
    "The Eiffel Tower is located in Berlin.",
    "Telemundo is an English-language television network.",
    "The Great Wall of China is visible from the Moon.",
]
picked = st.selectbox("Try an example, or type your own below:",
                      ["—"] + examples)

claim = st.text_input(
    "Claim to check",
    value="" if picked == "—" else picked,
    placeholder="e.g. The Eiffel Tower is located in Berlin.",
)

if st.button("Check claim", type="primary", disabled=not claim.strip()):
    verifier = get_verifier()
    with st.spinner("Retrieving evidence and reasoning…"):
        try:
            result = debunk(claim.strip(), verifier, max_web=max_web)
        except RuntimeError as err:
            st.error(f"{err}\n\nSet GOOGLE_FACTCHECK_KEY and TAVILY_KEY "
                     "(as Space secrets when deployed).")
            st.stop()

    st.markdown("### Verdict")
    verdict_card(result)

    if not result.evidence:
        st.info("No evidence was retrieved for this claim. This often happens "
                "for very obscure or very recent claims.")
    else:
        st.markdown("### Evidence")
        st.caption("Ordered by influence on the verdict. Filtered and neutral "
                   "evidence is shown dimmed for transparency.")
        for e in result.evidence:
            evidence_row(e)

st.markdown("---")
st.caption(
    "Built as a portfolio project on claim verification. Verdicts reflect "
    "retrieved evidence and can be wrong; not a substitute for professional "
    "fact-checking."
)