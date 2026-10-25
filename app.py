"""SOP Knowledge Copilot: Streamlit front end with a password gate and usage limits."""
import hmac
import os
import threading
import time
from datetime import datetime, timezone

import streamlit as st

from src.embed import load_env
from src.index import INDEX_NAME, get_client
from src.answer import answer_question, MIN_DENSE_SCORE

SESSION_LIMIT = 10   # questions per browser session
DAILY_LLM_CAP = 30   # model calls per day for the whole app (shares the Groq budget)
KEYS = ("GEMINI_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY")

st.set_page_config(page_title="SOP Knowledge Copilot")
load_env()


def secret(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


for _k in KEYS:
    _v = secret(_k)
    if _v:
        os.environ[_k] = _v


def esc(text: str) -> str:
    """Streamlit renders $...$ as math, so escape dollar signs."""
    return text.replace("$", "\\$")


@st.cache_resource
def get_usage() -> dict:
    return {"lock": threading.Lock(), "answer_lock": threading.Lock(), "day": "", "llm_calls": 0}


def llm_calls_today() -> int:
    u = get_usage()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with u["lock"]:
        if u["day"] != today:
            u["day"], u["llm_calls"] = today, 0
        return u["llm_calls"]


def add_llm_call() -> None:
    u = get_usage()
    with u["lock"]:
        u["llm_calls"] += 1


@st.cache_resource
def get_index_readonly():
    return get_client().Index(name=INDEX_NAME)


def check_password() -> bool:
    expected = secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD is not configured, so the app stays locked.")
        return False
    if st.session_state.get("authed"):
        return True
    st.title("SOP Knowledge Copilot")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode("utf-8"), expected.encode("utf-8")):
            st.session_state.authed = True
            st.rerun()
        else:
            time.sleep(1)
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

st.title("SOP Knowledge Copilot")
st.caption(
    "Ask about GitLab's Finance handbook (public snapshot, 2026-10-22). "
    "Answers cite their sources, and questions the handbook doesn't cover are declined."
)

if "count" not in st.session_state:
    st.session_state.count = 0
if "q" not in st.session_state:
    st.session_state.q = ""


def set_q(text: str) -> None:
    st.session_state.q = text


st.write("Try one:")
examples = [
    "How do I submit an expense report?",
    "What is the capital of France?",
    "How is payroll processed?",
]
for col, text in zip(st.columns(3), examples):
    col.button(text, on_click=set_q, args=(text,), use_container_width=True)

question = st.text_input("Your question", key="q")
left = SESSION_LIMIT - st.session_state.count
st.caption(f"Questions left in this session: {max(left, 0)}")

if st.button("Ask", type="primary") and question.strip():
    if left <= 0:
        st.warning("Session limit reached.")
    elif llm_calls_today() >= DAILY_LLM_CAP:
        st.warning("The daily demo limit has been reached. Please try again tomorrow (UTC).")
    else:
        st.session_state.count += 1
        res = None
        with st.spinner("Searching the handbook..."):
            with get_usage()["answer_lock"]:
                try:
                    res = answer_question(get_index_readonly(), question.strip())
                except (Exception, SystemExit) as exc:
                    st.error(f"Something went wrong ({type(exc).__name__}). Please try again later.")
        if res is not None:
            if res["reason"] != "low_similarity":
                add_llm_call()
            if res["refused"]:
                st.info(esc(res["answer"]))
            else:
                st.markdown(esc(res["answer"]))
            if res["sources"]:
                st.markdown("**Sources**")
                for s in res["sources"]:
                    st.markdown(f"- [{s['n']}] [{esc(s['heading_path'])}]({s['url']})")
            with st.expander("How this answer was decided"):
                outcome = f"declined ({res['reason']})" if res["refused"] else "answered"
                st.write(f"Top similarity score: {res['dense_top']:.3f} (needs at least {MIN_DENSE_SCORE}). Outcome: {outcome}.")
                for i, h in enumerate(res["hits"], start=1):
                    st.markdown(f"**[{i}]** {esc(h['heading_path'])}")
                    st.caption(esc(h["text"][:200].replace("\n", " ")) + "...")

st.divider()
st.caption(
    "Demo project, not affiliated with GitLab. Handbook content (c) GitLab, shown only as short "
    "excerpts with links to the source pages. Answers can be wrong, so check the sources."
)
