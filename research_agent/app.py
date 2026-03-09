import os
import re
import hashlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from services.pdf_parser import load_pdf
from services.text_splitter import split_text
from services.embedding_service import get_embedding_model
from services.vector_store import (
    chunks_to_documents,
    build_faiss_store,
    save_faiss_store,
    load_faiss_store,
)
from services.llm_service import get_llm
from services.analysis_service import answer_question_with_citations


REPORTS_DIR = Path("storage/reports")
INDEXES_DIR = Path("storage/indexes")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_answer_and_evidence(answer: str, docs: list) -> tuple:
    """Separate a clean short answer from any evidence the LLM echoed back."""
    snippets = {doc.page_content[:60] for doc in docs if doc.page_content}
    paragraphs = re.split(r"\n{2,}", answer.strip())
    clean, evidence = [], []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) > 200 and any(s in p for s in snippets):
            evidence.append(p)
        else:
            clean.append(p)
    return "\n\n".join(clean), evidence


def ensure_vector_store(pdf_path: str, index_dir: str):
    embeddings = get_embedding_model()
    store = load_faiss_store(embeddings, index_dir)
    if store is not None:
        return store
    text = load_pdf(pdf_path)
    chunks = split_text(text)
    docs = chunks_to_documents(chunks, source=pdf_path)
    store = build_faiss_store(docs, embeddings)
    save_faiss_store(store, index_dir)
    return store


# ── Neo-Brutalism SVG Characters ─────────────────────────────────────
CHARACTERS_SVG = """
<svg id="nb-eyes-svg" viewBox="-15 -30 330 310" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:300px;overflow:visible;">
  <!-- Orange rect -->
  <g transform="translate(5,10) rotate(-8, 55, 42)">
    <rect x="0" y="0" width="110" height="80" rx="10" fill="#FF9F43" stroke="#222" stroke-width="3.5"/>
    <circle cx="33" cy="32" r="9" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="77" cy="32" r="9" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="33" cy="32" r="4" fill="#222" class="nb-pupil" data-ox="33" data-oy="32" data-r="5"/>
    <circle cx="77" cy="32" r="4" fill="#222" class="nb-pupil" data-ox="77" data-oy="32" data-r="5"/>
  </g>
  <!-- Blue tall block -->
  <g transform="translate(125,-10) rotate(5, 50, 60)">
    <rect x="0" y="0" width="95" height="120" rx="10" fill="#7EB8DA" stroke="#222" stroke-width="3.5"/>
    <circle cx="30" cy="46" r="8" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="65" cy="46" r="8" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="30" cy="46" r="3.5" fill="#222" class="nb-pupil" data-ox="30" data-oy="46" data-r="4.5"/>
    <circle cx="65" cy="46" r="3.5" fill="#222" class="nb-pupil" data-ox="65" data-oy="46" data-r="4.5"/>
  </g>
  <!-- Red circle -->
  <g transform="translate(240,20) rotate(3)">
    <circle cx="0" cy="0" r="42" fill="#FF6B6B" stroke="#222" stroke-width="3.5"/>
    <circle cx="-14" cy="-10" r="8" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="14" cy="-10" r="8" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="-14" cy="-10" r="3.5" fill="#222" class="nb-pupil" data-ox="-14" data-oy="-10" data-r="4.5"/>
    <circle cx="14" cy="-10" r="3.5" fill="#222" class="nb-pupil" data-ox="14" data-oy="-10" data-r="4.5"/>
    <ellipse cx="1" cy="10" rx="8" ry="4.5" fill="#D94F4F"/>
  </g>
  <!-- Brown triangle -->
  <g transform="translate(-5,100) rotate(-10, 55, 50)">
    <polygon points="55,0 110,100 0,100" fill="#D4A574" stroke="#222" stroke-width="3.5"/>
    <circle cx="42" cy="62" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="68" cy="62" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="42" cy="62" r="3" fill="#222" class="nb-pupil" data-ox="42" data-oy="62" data-r="4"/>
    <circle cx="68" cy="62" r="3" fill="#222" class="nb-pupil" data-ox="68" data-oy="62" data-r="4"/>
  </g>
  <!-- Green square -->
  <g transform="translate(115,125) rotate(9, 40, 32)">
    <rect x="0" y="0" width="80" height="62" rx="8" fill="#95E77E" stroke="#222" stroke-width="3.5"/>
    <circle cx="25" cy="25" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="55" cy="25" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="25" cy="25" r="3" fill="#222" class="nb-pupil" data-ox="25" data-oy="25" data-r="4"/>
    <circle cx="55" cy="25" r="3" fill="#222" class="nb-pupil" data-ox="55" data-oy="25" data-r="4"/>
  </g>
  <!-- Pink wide rect -->
  <g transform="translate(195,118) rotate(-6, 52, 38)">
    <rect x="0" y="0" width="105" height="72" rx="10" fill="#FFB3BA" stroke="#222" stroke-width="3.5"/>
    <circle cx="33" cy="28" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="72" cy="28" r="7" fill="white" stroke="#222" stroke-width="2.5"/>
    <circle cx="33" cy="28" r="3" fill="#222" class="nb-pupil" data-ox="33" data-oy="28" data-r="4"/>
    <circle cx="72" cy="28" r="3" fill="#222" class="nb-pupil" data-ox="72" data-oy="28" data-r="4"/>
  </g>
  <!-- Yellow square -->
  <g transform="translate(60,205) rotate(6, 38, 30)">
    <rect x="0" y="0" width="72" height="55" rx="8" fill="#FFE66D" stroke="#222" stroke-width="3.5"/>
    <circle cx="22" cy="22" r="6.5" fill="white" stroke="#222" stroke-width="2"/>
    <circle cx="50" cy="22" r="6.5" fill="white" stroke="#222" stroke-width="2"/>
    <circle cx="22" cy="22" r="3" fill="#222" class="nb-pupil" data-ox="22" data-oy="22" data-r="3.5"/>
    <circle cx="50" cy="22" r="3" fill="#222" class="nb-pupil" data-ox="50" data-oy="22" data-r="3.5"/>
  </g>
  <!-- Small blue block -->
  <g transform="translate(-5,210) rotate(-12, 30, 24)">
    <rect x="0" y="0" width="58" height="48" rx="7" fill="#7EB8DA" stroke="#222" stroke-width="3.5"/>
    <circle cx="18" cy="19" r="6" fill="white" stroke="#222" stroke-width="2"/>
    <circle cx="40" cy="19" r="6" fill="white" stroke="#222" stroke-width="2"/>
    <circle cx="18" cy="19" r="2.5" fill="#222" class="nb-pupil" data-ox="18" data-oy="19" data-r="3.5"/>
    <circle cx="40" cy="19" r="2.5" fill="#222" class="nb-pupil" data-ox="40" data-oy="19" data-r="3.5"/>
  </g>
</svg>
"""

# ── Eye-tracking + toolbar-hide JS ────────────────────────────────────
INTERACTIVE_JS = """
<script>
(function(){
  /* ── Eye tracking ── */
  function bootEyes(){
    var svg = document.getElementById('nb-eyes-svg');
    if(!svg){setTimeout(bootEyes,400);return;}
    var pupils = svg.querySelectorAll('.nb-pupil');
    if(!pupils.length){setTimeout(bootEyes,400);return;}

    document.addEventListener('mousemove',function(e){
      requestAnimationFrame(function(){
        var r = svg.getBoundingClientRect();
        var vbX=-15, vbY=-30, vbW=330, vbH=310;

        pupils.forEach(function(p){
          var ox=+p.dataset.ox, oy=+p.dataset.oy, maxR=+p.dataset.r;
          var g=p.closest('g');
          var pt=svg.createSVGPoint(); pt.x=ox; pt.y=oy;
          var ctm=g.getCTM(); if(!ctm)return;
          var sp=pt.matrixTransform(ctm);
          var sx=r.left+(sp.x-vbX)/vbW*r.width;
          var sy=r.top +(sp.y-vbY)/vbH*r.height;
          var dx=e.clientX-sx, dy=e.clientY-sy;
          var d=Math.sqrt(dx*dx+dy*dy)||1;
          var s=Math.min(maxR/d,1);
          p.setAttribute('cx', ox+dx*s*0.18);
          p.setAttribute('cy', oy+dy*s*0.18);
        });
      });
    },{passive:true});

    document.addEventListener('mouseleave',function(){
      pupils.forEach(function(p){
        p.setAttribute('cx',p.dataset.ox);
        p.setAttribute('cy',p.dataset.oy);
      });
    });
  }
  bootEyes();

  /* ── Toolbar scroll-hide ── */
  var lastY=0;
  var chk=setInterval(function(){
    var tb=parent.document.querySelector('[data-testid="stToolbar"]');
    if(!tb)return; clearInterval(chk);
    window.addEventListener('scroll',function(){
      var y=window.scrollY||document.documentElement.scrollTop;
      if(y>60&&y>lastY){tb.style.opacity='0';tb.style.pointerEvents='none';tb.style.transform='translateY(-20px)';}
      else{tb.style.opacity='1';tb.style.pointerEvents='auto';tb.style.transform='translateY(0)';}
      lastY=y;
    },{passive:true});
  },300);
})();
</script>
"""

# ── Neo-Brutalism CSS ─────────────────────────────────────────────────
NEOBRUTALISM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --nb-black: #222222;
    --nb-white: #FAFAFA;
    --nb-bg: #F5F0EB;
    --nb-orange: #FF9F43;
    --nb-blue: #7EB8DA;
    --nb-red: #FF6B6B;
    --nb-green: #95E77E;
    --nb-yellow: #FFE66D;
    --nb-pink: #FFB3BA;
    --nb-shadow: 4px 4px 0 var(--nb-black);
    --nb-shadow-sm: 3px 3px 0 var(--nb-black);
    --nb-border: 3px solid var(--nb-black);
    --nb-radius: 12px;
    --nb-font: 'Space Grotesk', system-ui, sans-serif;
}

/* ── Global – force bg color on ALL containers ── */
html, body,
.stApp, .stApp > *,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > *,
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > *,
[data-testid="stMain"], [data-testid="stMain"] > *,
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"],
.stChatFloatingInputContainer,
.block-container, footer {
    background: var(--nb-bg) !important;
    background-color: var(--nb-bg) !important;
}
.stApp { font-family: var(--nb-font) !important; }
.stApp * { font-family: var(--nb-font) !important; }

/* ── Hide sidebar completely ── */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="headerNoPadding"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
}

/* Hide default Streamlit header chrome */
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Expand main to full width ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    max-width: 840px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Leave space at bottom for the floating chat input */
.main .block-container {
    padding-bottom: 100px !important;
}

/* ── Neo-Brutalism Header Card ── */
.nb-header {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 2rem;
    margin-bottom: 1.5rem;
    background: var(--nb-white);
    border: var(--nb-border);
    border-radius: var(--nb-radius);
    box-shadow: var(--nb-shadow);
}
.nb-header-text h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--nb-black) !important;
    margin: 0 0 0.3rem 0 !important;
    line-height: 1.2 !important;
}
.nb-header-text p {
    font-size: 1rem !important;
    color: #666 !important;
    margin: 0 !important;
    font-weight: 400 !important;
}
.nb-chars { flex-shrink: 0; width: 260px; }

/* ── Status pills ── */
.nb-pills {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.nb-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border: 2px solid var(--nb-black);
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    box-shadow: 2px 2px 0 var(--nb-black);
}
.nb-pill-orange { background: var(--nb-orange); }
.nb-pill-blue { background: var(--nb-blue); }
.nb-pill-green { background: var(--nb-green); }

/* ── Info card (pink) ── */
.nb-card-pink {
    background: var(--nb-pink);
    border: var(--nb-border);
    border-radius: var(--nb-radius);
    box-shadow: var(--nb-shadow-sm);
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.nb-card-pink h3 {
    font-size: 0.9rem !important; font-weight: 700 !important;
    margin: 0 0 0.2rem 0 !important; color: var(--nb-black) !important;
}
.nb-card-pink p {
    font-size: 0.85rem !important; margin: 0 !important; color: #444 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--nb-white) !important;
    border: var(--nb-border) !important;
    border-radius: var(--nb-radius) !important;
    box-shadow: var(--nb-shadow) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: var(--nb-black) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--nb-yellow) !important;
    color: var(--nb-black) !important;
    border: var(--nb-border) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: var(--nb-shadow-sm) !important;
    transition: all 0.1s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translate(1px, 1px) !important;
    box-shadow: 2px 2px 0 var(--nb-black) !important;
}
.stButton > button:active {
    transform: translate(3px, 3px) !important;
    box-shadow: 0 0 0 var(--nb-black) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    border: var(--nb-border) !important;
    border-radius: var(--nb-radius) !important;
    box-shadow: var(--nb-shadow-sm) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 1rem !important;
    background: var(--nb-white) !important;
}
/* Hide Material-icon text fallbacks everywhere */
.material-symbols-outlined,
.material-symbols-rounded,
.material-icons {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    user-select: none !important;
}
/* Keep expander arrow clickable (let the text label do the job) */
[data-testid="stExpanderToggleIcon"] svg {
    display: inline-block !important;
    width: 1rem !important;
    height: 1rem !important;
}

/* ── Floating Chat Input at bottom ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 1.2rem !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(800px, calc(100% - 4rem)) !important;
    z-index: 999 !important;
    border: var(--nb-border) !important;
    border-radius: var(--nb-radius) !important;
    box-shadow: var(--nb-shadow), 0 -4px 20px rgba(0,0,0,0.08) !important;
    background: var(--nb-white) !important;
    padding: 0.5rem !important;
}

/* ── Hide Deploy button on scroll ── */
[data-testid="stToolbar"] {
    transition: opacity 0.3s ease, transform 0.3s ease !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--nb-white) !important;
    border: var(--nb-border) !important;
    border-radius: var(--nb-radius) !important;
    box-shadow: var(--nb-shadow-sm) !important;
}

/* ── Alerts ── */
.stAlert {
    border: var(--nb-border) !important;
    border-radius: var(--nb-radius) !important;
    box-shadow: var(--nb-shadow-sm) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 2px dashed #ccc !important;
    margin: 1.5rem 0 !important;
}

/* ── Fill ALL bottom / side areas with bg color ── */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > *,
[data-testid="stBottom"],
[data-testid="stBottom"] > *,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > *,
[data-testid="stAppViewBlockContainer"],
footer, .main > div:last-child,
.stApp > div,
.stApp > section,
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > div {
    background: var(--nb-bg) !important;
    background-color: var(--nb-bg) !important;
}

/* Spinner text color */
.stSpinner { color: var(--nb-black) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--nb-bg); }
::-webkit-scrollbar-thumb { background: var(--nb-black); border-radius: 4px; }
</style>
"""

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Analyzer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(NEOBRUTALISM_CSS, unsafe_allow_html=True)

# Eye-tracking + toolbar scroll-hide JS
st.markdown(INTERACTIVE_JS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="nb-header">
        <div class="nb-chars">{CHARACTERS_SVG}</div>
        <div class="nb-header-text">
            <h1>Cortis Analyzer</h1>
            <p>Upload a PDF research report and ask questions. AI-powered analysis at your fingertips.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session State ─────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "index_dir" not in st.session_state:
    st.session_state.index_dir = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Upload Section ────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a PDF research report", type=["pdf"])

if uploaded is not None:
    pdf_bytes = uploaded.read()
    file_hash = sha256_bytes(pdf_bytes)
    save_name = f"{file_hash}_{uploaded.name}"
    pdf_path = REPORTS_DIR / save_name
    index_dir = INDEXES_DIR / file_hash

    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)

    st.session_state.pdf_path = str(pdf_path)
    st.session_state.index_dir = str(index_dir)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Build / Load Index", type="primary"):
            with st.spinner("Building vector index..."):
                st.session_state.store = ensure_vector_store(
                    st.session_state.pdf_path,
                    st.session_state.index_dir,
                )
            st.success("Index ready! Ask me anything.")
    with col2:
        st.markdown(
            f'<div class="nb-card-pink">'
            f'<h3>Current file</h3><p>{uploaded.name}</p></div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Upload a PDF above and click **Build / Load Index** to get started.")

st.divider()

# ── Chat History ──────────────────────────────────────────────────────
_AVATARS = {"user": "👤", "assistant": "⭐"}
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=_AVATARS.get(m["role"])):
        st.write(m["content"])

# ── Chat Input (rendered as floating by CSS) ──────────────────────────
question = st.chat_input("Ask a question about the report...")

if question:
    if st.session_state.store is None:
        st.warning("Please upload a PDF and click **Build / Load Index** first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.write(question)

        with st.chat_message("assistant", avatar="⭐"):
            with st.spinner("Searching evidence + generating answer..."):
                try:
                    llm = get_llm(temperature=0)
                    answer, cited_ids, docs = answer_question_with_citations(
                        st.session_state.store, llm, question, k=4
                    )
                except Exception as e:
                    answer = f"Error: {e}"
                    cited_ids, docs = [], []

            clean_answer, _ = _split_answer_and_evidence(answer, docs)
            display_text = clean_answer if clean_answer else answer
            st.write(display_text)

            if docs:
                with st.expander("View evidence chunks"):
                    for i, d in enumerate(docs, start=1):
                        tag = " 🔖" if i in cited_ids else ""
                        st.markdown(f"**[{i}]{tag}**")
                        st.write(d.page_content[:600])
                        st.markdown("---")

        st.session_state.messages.append({"role": "assistant", "content": display_text})
