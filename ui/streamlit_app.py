import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="HR Policy Assistant", layout="centered")
st.markdown(
    """<style>
    blockquote {border-left: 3px solid #7c9cff; background: #161a22; padding: .5rem .9rem; margin: .4rem 0;}
    blockquote p {color: #cfd5e3;}
    [data-testid="stExpander"] details {border-color: #2a2f3a;}
    </style>""",
    unsafe_allow_html=True,
)
st.title("HR Policy Assistant")
st.caption("Answers come only from uploaded HR policies, with citations. If the policies don't say, it tells you so.")

with st.sidebar:
    st.header("Policies")
    try:
        docs = httpx.get(f"{API}/documents", timeout=10).json()
    except httpx.HTTPError:
        st.error(f"API not reachable at {API}")
        st.stop()
    for d in docs:
        st.markdown(f"**{d['title']}**  \n`{d['filename']}` · {d['chunks']} chunks · v{d['version']}")
    if not docs:
        st.info("No policies uploaded yet.")

    st.divider()
    st.subheader("Admin: upload")
    token = st.text_input("Admin token", type="password")
    files = st.file_uploader("Policy files", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if st.button("Upload", disabled=not files):
        for f in files:
            r = httpx.post(f"{API}/documents", headers={"X-Admin-Token": token},
                           files={"file": (f.name, f.getvalue())}, timeout=120)
            if r.status_code == 200:
                st.success(f"{f.name}: {r.json()['status']} ({r.json()['chunks']} chunks)")
            else:
                st.error(f"{f.name}: {r.json().get('detail', r.text)}")
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


def render(resp: dict):
    if resp["status"] == "refused":
        st.warning(resp["answer"])
        st.caption(f"refusal_reason: `{resp['refusal_reason']}`")
        return
    st.markdown(resp["answer"])
    for c in resp["citations"]:
        with st.expander(f"{c['document']} · {c['section']}"):
            st.markdown(f"> {c['quote']}")
            st.caption(f"{c['chunk_id']} · version {c['doc_version']}")


for q, resp in st.session_state.history:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        render(resp)

if q := st.chat_input("Ask about leave, benefits, IT policy…"):
    st.chat_message("user").write(q)
    with st.chat_message("assistant"), st.spinner("Checking the policies…"):
        r = httpx.post(f"{API}/ask", json={"question": q}, timeout=60)
        resp = r.json() if r.status_code == 200 else {
            "status": "refused", "answer": "Please type a question.", "refusal_reason": "invalid_request"}
        render(resp)
    st.session_state.history.append((q, resp))
    with st.expander("Raw JSON"):
        st.json(resp)
