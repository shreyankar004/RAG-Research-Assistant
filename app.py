import streamlit as st
import streamlit.components.v1 as components

from agents.chat import ChatAgent
from agents.chunking import ChunkingAgent
from agents.embeddings import EmbeddingAgent
from agents.pdf_reader import PDFReaderAgent
from agents.planner import PlannerAgent
from agents.research import ResearchAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.vector_store import VectorStoreAgent
from agents.web_search import WebSearchAgent
from orchestration import AGENT_ROLES
from utils.report import build_research_pdf

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html,body,*{font-family:'Manrope','Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;box-sizing:border-box;}

:root{
  --bg:#F6F7FB;
  --white:#FFFFFF;
  --card:#FFFFFF;
  --border:#E4E7FF;
  --border2:#C7CEFF;
  --text:#111827;
  --sub:#6B7280;
  --muted:#9CA3AF;
  --brand:#2563EB;
  --brand-light:#EEF2FF;
  --brand-mid:#C7D2FE;
  --brand-dark:#1E40AF;
  --green:#059669;
  --green-bg:#ECFDF5;
  --green-bd:#6EE7B7;
  --red:#DC2626;
  --red-bg:#FEF2F2;
  --purple:#0F766E;
  --purple-bg:#F5F3FF;
  --amber:#D97706;
  --shadow-sm:0 1px 3px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
  --shadow:0 4px 16px rgba(79,70,229,0.08),0 1px 4px rgba(0,0,0,0.04);
  --shadow-lg:0 10px 40px rgba(79,70,229,0.12),0 2px 8px rgba(0,0,0,0.06);
}

/* ── BASE ── */
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;visibility:hidden!important;}

/* ── NAV ── */
.rm-nav{
  background:var(--white);
  border-bottom:1px solid var(--border);
  padding:0 32px;height:60px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:1000;
  box-shadow:var(--shadow-sm);
}
.rm-logo{
  width:36px;height:36px;
  background:linear-gradient(135deg,#4F46E5,#7C3AED);
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:14px;font-weight:900;color:#fff;letter-spacing:-0.5px;
  box-shadow:0 2px 8px rgba(79,70,229,0.35);
}
.rm-brand{display:flex;align-items:center;gap:12px;}
.rm-title{font-size:1.18rem;font-weight:900;color:var(--text);letter-spacing:0;}
.rm-sub{font-size:0.76rem;color:var(--sub);font-weight:600;margin-top:1px;}
.rm-badge{
  display:flex;align-items:center;gap:6px;
  background:var(--brand-light);border:1px solid var(--brand-mid);
  border-radius:20px;padding:4px 12px;
}
.rm-badge-dot{width:7px;height:7px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 2px var(--green-bg);}
.rm-badge-text{font-size:0.7rem;font-weight:700;color:var(--brand);}

/* ── MODE SWITCHER ── */
.rm-modes{
  background:var(--white);border-bottom:1px solid var(--border);
  padding:0 32px;
}
/* Hide Streamlit radio default rendering */
.rm-modes [data-testid="stRadio"] > label{display:none!important;}
.rm-modes [data-baseweb="radio-group"]{
  display:flex!important;flex-direction:row!important;gap:0!important;
  background:transparent!important;
}
.rm-modes [data-baseweb="radio"]{
  padding:0 20px!important;height:46px!important;
  display:flex!important;align-items:center!important;
  cursor:pointer!important;position:relative!important;
  border-bottom:2.5px solid transparent!important;
  transition:all 0.15s!important;
}
.rm-modes [data-baseweb="radio"]:has(input:checked){
  border-bottom-color:var(--brand)!important;
}
.rm-modes [data-baseweb="radio"] [data-testid="stMarkdownContainer"] p{
  font-size:0.84rem!important;font-weight:600!important;
  color:var(--sub)!important;white-space:nowrap!important;
}
.rm-modes [data-baseweb="radio"]:has(input:checked) [data-testid="stMarkdownContainer"] p{
  color:var(--brand)!important;
}
/* Hide radio dot */
.rm-modes [data-baseweb="radio"] > div:first-child{display:none!important;}
.rm-modes [data-testid="stWidgetLabel"]{display:none!important;}
.rm-modes .stRadio{padding:0!important;margin:0!important;}

/* ── PAGE TABS ── */
div[data-testid="stTabs"] [data-baseweb="tab-list"]{
  background:var(--white)!important;
  border-bottom:1px solid var(--border)!important;
  padding:0 32px!important;gap:0!important;
  box-shadow:none!important;
}
div[data-testid="stTabs"] [data-baseweb="tab"]{
  background:transparent!important;color:var(--sub)!important;
  border:none!important;font-weight:600!important;font-size:0.82rem!important;
  padding:10px 16px!important;border-bottom:2px solid transparent!important;
  border-radius:0!important;transition:all 0.15s!important;
}
div[data-testid="stTabs"] [aria-selected="true"]{
  color:var(--text)!important;border-bottom-color:var(--brand)!important;
}
div[data-testid="stTabs"] [data-baseweb="tab-panel"]{
  padding:24px 32px!important;background:transparent!important;
}

/* ── CARDS ── */
.rm-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;padding:18px 20px;margin-bottom:14px;
  box-shadow:var(--shadow-sm);
}
.rm-card-title{
  font-size:0.72rem;font-weight:900;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;
  display:flex;align-items:center;gap:6px;
}

/* ── CUSTOM UPLOAD DROP ZONE ── */
.rm-upload-zone{
  border:1.5px dashed var(--brand-mid);border-radius:12px;
  padding:28px 20px;text-align:center;background:var(--brand-light);
  transition:all 0.2s;cursor:pointer;position:relative;
}
.rm-upload-zone:hover{
  border-color:var(--brand);background:#E8EAFF;
  box-shadow:0 0 0 4px rgba(79,70,229,0.08);
}
.rm-upload-icon{font-size:2rem;margin-bottom:8px;display:block;}
.rm-upload-title{font-size:0.88rem;font-weight:700;color:var(--brand);margin-bottom:4px;}
.rm-upload-sub{font-size:0.72rem;color:var(--sub);}

/* ── FILE UPLOADER — real functional styling ── */
[data-testid="stFileUploader"] label [data-testid="stMarkdownContainer"] p{
  font-size:0.82rem!important;font-weight:700!important;color:var(--brand)!important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"]{
  background:var(--brand-light)!important;
  border:1.5px dashed var(--brand-mid)!important;
  border-radius:12px!important;
  padding:24px 20px!important;
  text-align:center!important;
  transition:all 0.2s!important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploadDropzone"]:hover{
  border-color:var(--brand)!important;
  background:#E4E8FF!important;
  box-shadow:0 0 0 4px rgba(79,70,229,0.1)!important;
}
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploaderDropzone"] button{
  background:var(--brand)!important;color:#fff!important;
  border-radius:8px!important;border:none!important;
  font-weight:700!important;font-size:0.78rem!important;
  padding:6px 16px!important;margin-top:8px!important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploaderDropzone"] span{
  color:var(--sub)!important;font-size:0.78rem!important;
}
.rm-upload-zone{display:none!important;}

/* ── FILE BADGE ── */
.rm-file-badge{
  background:var(--green-bg);border:1.5px solid var(--green-bd);
  border-radius:10px;padding:10px 14px;margin-top:10px;
  display:flex;align-items:center;gap:10px;
}
.rm-file-icon{font-size:1.3rem;}
.rm-file-name{font-size:0.8rem;font-weight:700;color:var(--green);}
.rm-file-size{font-size:0.68rem;color:var(--sub);}

/* ── METRICS ── */
.rm-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;}
.rm-metric{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;box-shadow:var(--shadow-sm);
}
.rm-metric-label{font-size:0.6rem;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;}
.rm-metric-val{font-size:1.4rem;font-weight:900;color:var(--text);letter-spacing:-0.5px;}
.rm-metric-sub{font-size:0.65rem;color:var(--sub);margin-top:2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* ── PIPELINE ── */
.rm-pipeline{display:flex;align-items:center;flex-wrap:wrap;gap:4px;padding:6px 0;}
.rm-pnode{
  display:flex;align-items:center;gap:5px;padding:5px 11px;
  border-radius:20px;border:1px solid var(--border);
  background:var(--bg);font-size:0.68rem;font-weight:600;
  white-space:nowrap;color:var(--sub);transition:all 0.2s;
}
.rm-pnode.done{border-color:var(--green-bd);background:var(--green-bg);color:var(--green);}
.rm-pnode.run{border-color:var(--brand-mid);background:var(--brand-light);color:var(--brand);}
.rm-pnode.err{border-color:#FCA5A5;background:var(--red-bg);color:var(--red);}
.rm-pdot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.done .rm-pdot{background:var(--green);box-shadow:0 0 0 3px rgba(5,150,105,0.15);}
.run .rm-pdot{background:var(--brand);animation:pulse 1.2s infinite;}
.err .rm-pdot{background:var(--red);}
.rm-pnode.wait .rm-pdot{background:var(--border2);}
.rm-parr{color:var(--border2);padding:0 2px;font-size:0.72rem;font-weight:700;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.5;transform:scale(0.85);}}

/* ── REPORT SECTIONS ── */
.rm-section{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:16px 18px;margin-bottom:10px;
  box-shadow:var(--shadow-sm);transition:box-shadow 0.2s;
}
.rm-section:hover{box-shadow:var(--shadow);}
.rm-section-tag{
  font-size:0.6rem;font-weight:800;color:var(--brand);
  text-transform:uppercase;letter-spacing:0.09em;
  padding:3px 9px;background:var(--brand-light);border-radius:20px;
  display:inline-block;margin-bottom:9px;
}
.rm-section-text{font-size:0.98rem;color:var(--text);line-height:1.78;}

/* ── SUMMARY ── */
.rm-summary{
  background:linear-gradient(135deg,var(--green-bg),#D1FAE5);
  border:1.5px solid var(--green-bd);border-radius:12px;
  padding:16px 18px;margin-bottom:14px;
}
.rm-summary-label{
  font-size:0.6rem;font-weight:800;color:var(--green);
  text-transform:uppercase;letter-spacing:0.09em;margin-bottom:8px;
  display:flex;align-items:center;gap:6px;
}
.rm-summary-text{font-size:1rem;color:var(--text);line-height:1.78;}

/* ── REFERENCES ── */
.rm-ref{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;margin-bottom:8px;display:flex;gap:12px;
  box-shadow:var(--shadow-sm);
}
.rm-ref-num{
  width:28px;height:28px;border-radius:50%;background:var(--brand-light);
  color:var(--brand);font-size:0.7rem;font-weight:800;display:flex;
  align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;
}
.rm-ref-title{font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:3px;}
.rm-ref-url{font-size:0.7rem;color:var(--brand);margin-bottom:5px;word-break:break-all;}
.rm-ref-snip{font-size:0.75rem;color:var(--sub);line-height:1.55;}
.rm-ref-score{
  font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:20px;
  background:var(--brand-light);color:var(--brand);white-space:nowrap;
}

/* ── CHAT ── */
.rm-chat{
  background:var(--card);border:1px solid var(--border);
  border-radius:14px;overflow:hidden;box-shadow:var(--shadow);
}
.rm-chat-head{
  background:linear-gradient(135deg,var(--brand),var(--purple));
  padding:12px 16px;display:flex;align-items:center;gap:10px;
}
.rm-chat-head-dot{width:8px;height:8px;border-radius:50%;background:#fff;opacity:0.7;}
.rm-chat-head-label{font-size:0.78rem;font-weight:700;color:#fff;}
.rm-chat-body{padding:16px;display:flex;flex-direction:column;gap:12px;min-height:200px;}
.rm-msg{display:flex;gap:10px;}
.rm-av{
  width:30px;height:30px;border-radius:9px;display:flex;
  align-items:center;justify-content:center;font-size:0.65rem;
  font-weight:800;flex-shrink:0;
}
.rm-av.u{background:var(--brand-light);color:var(--brand);}
.rm-av.a{background:var(--purple-bg);color:var(--purple);}
.rm-msg-body{flex:1;}
.rm-msg-role{font-size:0.6rem;font-weight:800;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;}
.rm-msg-text{font-size:0.96rem;color:var(--text);line-height:1.7;
  background:var(--bg);padding:9px 13px;border-radius:10px;}
.rm-msg.u .rm-msg-text{background:var(--brand-light);color:var(--text);}

/* ── AGENT CARDS ── */
.rm-agent{
  display:flex;align-items:flex-start;gap:12px;
  background:var(--bg);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px;margin-bottom:7px;
  transition:all 0.15s;
}
.rm-agent:hover{background:var(--brand-light);border-color:var(--brand-mid);}
.rm-agent-ic{
  width:34px;height:34px;border-radius:9px;background:var(--brand-light);
  display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;
}
.rm-agent-name{font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:3px;}
.rm-agent-goal{font-size:0.72rem;color:var(--sub);line-height:1.5;}

/* ── EMPTY STATE ── */
.rm-empty{
  text-align:center;padding:48px 20px;background:var(--card);
  border:1.5px dashed var(--border);border-radius:14px;margin:10px 0;
}
.rm-empty-icon{font-size:2.4rem;margin-bottom:12px;display:block;}
.rm-empty-title{font-size:1rem;font-weight:700;color:var(--text);margin-bottom:6px;}
.rm-empty-desc{font-size:0.8rem;color:var(--sub);}

/* ── SOURCE ITEMS ── */
.rm-src{padding:7px 0;border-bottom:1px solid var(--border);}
.rm-src:last-child{border-bottom:none;}
.rm-src-title{font-size:0.76rem;font-weight:600;color:var(--brand);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.rm-src-url{font-size:0.63rem;color:var(--sub);}

/* ── STREAMLIT WIDGETS ── */
.stTextInput input,.stTextArea textarea{
  background:var(--bg)!important;color:var(--text)!important;
  border:1.5px solid var(--border)!important;border-radius:9px!important;
  font-size:0.96rem!important;padding:11px 13px!important;
  box-shadow:none!important;transition:border-color 0.15s,box-shadow 0.15s!important;
}
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color:var(--brand)!important;
  box-shadow:0 0 0 3px rgba(79,70,229,0.12)!important;
}
[data-testid="stWidgetLabel"] p{
  color:var(--sub)!important;font-size:0.65rem!important;
  font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.07em!important;
}
.stButton>button{
  border-radius:9px!important;font-weight:800!important;font-size:0.94rem!important;
  height:44px!important;border:none!important;transition:all 0.18s!important;
  letter-spacing:-0.1px!important;
}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,var(--brand),var(--brand-dark))!important;
  color:#fff!important;box-shadow:0 2px 8px rgba(79,70,229,0.3)!important;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 4px 16px rgba(79,70,229,0.4)!important;transform:translateY(-1px)!important;
}
.stButton>button[kind="secondary"]{
  background:var(--card)!important;border:1.5px solid var(--border)!important;
  color:var(--text)!important;
}
.stButton>button[kind="secondary"]:hover{background:var(--bg)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--brand),var(--purple))!important;border-radius:4px!important;}
[data-testid="stExpander"]{border:none!important;background:transparent!important;}
[data-testid="stExpander"] details{border:1.5px solid var(--border)!important;border-radius:10px!important;background:var(--card)!important;}
[data-testid="stExpander"] details summary{list-style:none!important;padding:10px 14px!important;font-weight:600!important;font-size:0.82rem!important;color:var(--text)!important;cursor:pointer!important;}
[data-testid="stExpander"] details summary::-webkit-details-marker{display:none!important;}
[data-testid="stExpander"] details summary::marker{content:""!important;}
[data-testid="stExpanderToggleIcon"]{display:none!important;}
.stAlert{border-radius:10px!important;}
.stDownloadButton>button{
  background:var(--green-bg)!important;border:1.5px solid var(--green-bd)!important;
  color:var(--green)!important;border-radius:9px!important;font-weight:700!important;
}
</style>
""", unsafe_allow_html=True)


# ── STATE ──────────────────────────────────────────────────────────────────────
def _init():
    d = dict(mode="pdf",pdf_text="",chunks=[],vector_store=None,document_key=None,
             last_pdf_name="",topic="",top_k=5,plan="",research={},summary="",
             chat_history=[],agent_status={},last_context="",web_results=[],
             web_sources=[],general_chunks=[],general_vector_store=None)
    for k,v in d.items(): st.session_state.setdefault(k,v)

def _set(k,s): st.session_state.agent_status[k]=s

def _cls(s):
    s=s.lower()
    if "done" in s or "ready" in s: return "done"
    if "run" in s: return "run"
    if "error" in s: return "err"
    return "wait"


# ── AGENTS ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agents():
    return dict(reader=PDFReaderAgent(),chunking=ChunkingAgent(),
                embedding=EmbeddingAgent(),vector_store=VectorStoreAgent(),
                retriever=RetrieverAgent(),planner=PlannerAgent(),
                research=ResearchAgent(),summarizer=SummarizerAgent(),
                chat=ChatAgent(),web_search=WebSearchAgent())


# ── PIPELINE STEPS ─────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    ag=get_agents()
    try:
        prog=st.progress(0)
        _set("PDF Reader","Running"); prog.progress(10,"Extracting text…")
        r=ag["reader"].extract(uploaded_file)
        st.session_state.pdf_text=r.text
        st.session_state.document_key=r.document_key
        st.session_state.last_pdf_name=uploaded_file.name
        _set("PDF Reader",f"Done ({r.page_count} pages)")
        _set("Chunking","Running"); prog.progress(30,"Chunking document…")
        chunks=ag["chunking"].chunk(r.text)
        st.session_state.chunks=chunks
        _set("Chunking",f"Done ({len(chunks)} chunks)")
        _set("Embedding","Running"); prog.progress(55,"Generating embeddings…")
        embeddings=ag["embedding"].embed_documents([c.text for c in chunks])
        _set("Embedding","Done")
        _set("Vector Store","Running"); prog.progress(80,"Building vector index…")
        vs=ag["vector_store"].build_or_load(r.document_key,chunks,embeddings)
        st.session_state.vector_store=vs
        _set("Vector Store",f"Ready ({vs.ntotal} vectors)")
        prog.progress(100,"✅ Ready!")
        return True
    except Exception as e:
        st.error(f"PDF error: {e}"); _set("PDF Reader","Error"); return False

def run_web_search(topic):
    ag=get_agents()
    _set("Web Search","Running")
    r=ag["web_search"].search(topic,max_results=8)
    if not r.ok: _set("Web Search","Error"); st.error(r.error); return False
    st.session_state.web_results=r.results
    st.session_state.web_sources=[{"title":x.title,"url":x.url,"content":x.content,"score":x.score} for x in r.results]
    st.session_state.last_context=r.combined_context
    _set("Web Search",f"Done ({len(r.results)} sources)")
    _set("Retriever","Running")
    chunks=ag["chunking"].chunk(r.combined_context)
    st.session_state.general_chunks=chunks
    if chunks:
        import hashlib
        embs=ag["embedding"].embed_documents([c.text for c in chunks])
        key=hashlib.md5(topic.encode()).hexdigest()+"_web"
        vs=ag["vector_store"].build_or_load(key,chunks,embs)
        st.session_state.general_vector_store=vs
    _set("Retriever","Done"); return True

def retrieve_ctx(query,top_k,mode):
    ag=get_agents()
    vs=st.session_state.vector_store if mode=="pdf" else st.session_state.general_vector_store
    if not vs: return st.session_state.last_context[:6000]
    _set("Retriever","Running")
    q_emb=ag["embedding"].embed_query(query, vs.vocabulary, vs.idf)
    chunks=ag["retriever"].retrieve(vs,q_emb,top_k=top_k)
    ctx="\n\n".join(c.text for c in chunks)
    st.session_state.last_context=ctx
    _set("Retriever","Done")
    return ctx

def create_plan():
    ag=get_agents(); topic=st.session_state.topic; mode=st.session_state.mode
    if not topic: st.warning("Enter a research topic first."); return False
    _set("Planner","Running")
    ctx=retrieve_ctx(topic,st.session_state.top_k,mode)
    r=ag["planner"].create_plan(topic=topic,context=ctx,mode=mode)
    if r.ok: st.session_state.plan=r.text; _set("Planner","Done"); return True
    _set("Planner","Error"); st.error(r.text); return False

def run_research():
    ag=get_agents(); topic=st.session_state.topic; mode=st.session_state.mode
    if not topic: st.warning("Enter a research topic first."); return False
    ctx=retrieve_ctx(topic,st.session_state.top_k,mode)
    _set("Research","Running")
    res=ag["research"].research(topic=topic,plan=st.session_state.plan,context=ctx,mode=mode)
    if not res.ok: _set("Research","Error"); st.error(res.error); return False
    st.session_state.research=res.sections; _set("Research","Done")
    _set("Summarizer","Running")
    summ=ag["summarizer"].summarize(topic=topic,research=res.sections,context=ctx)
    if summ.ok: st.session_state.summary=summ.text; _set("Summarizer","Done")
    else: _set("Summarizer","Error")
    return True

def run_full(uploaded_file=None):
    mode=st.session_state.mode
    if not st.session_state.topic: st.warning("Enter a research topic first."); return
    with st.status("Running pipeline…",expanded=True) as s:
        if mode=="pdf":
            if not uploaded_file: st.warning("Upload a PDF first."); return
            st.write("📄 Processing PDF…")
            if not process_pdf(uploaded_file): return
        else:
            st.write("🌐 Searching the web…")
            if not run_web_search(st.session_state.topic): return
        st.write("📋 Creating research plan…"); create_plan()
        st.write("✍️ Generating report…"); run_research()
        s.update(label="✅ Complete!",state="complete")


# ── PIPELINE BAR ───────────────────────────────────────────────────────────────
def _pipeline_bar():
    mode=st.session_state.mode
    nodes=(["PDF Reader","Chunking","Embedding","Vector Store","Retriever","Planner","Research","Summarizer"]
           if mode=="pdf" else ["Web Search","Retriever","Planner","Research","Summarizer"])
    parts=[]
    for i,n in enumerate(nodes):
        stat=st.session_state.agent_status.get(n,"Waiting")
        parts.append(f'<div class="rm-pnode {_cls(stat)}"><div class="rm-pdot"></div>{n}</div>')
        if i<len(nodes)-1: parts.append('<span class="rm-parr">›</span>')
    st.markdown(f'<div class="rm-pipeline">{"".join(parts)}</div>',unsafe_allow_html=True)


# ── REPORT RENDERER ────────────────────────────────────────────────────────────
def _render_report(mode):
    if st.session_state.plan:
        with st.expander("📋 View Research Plan",expanded=False):
            st.markdown(st.session_state.plan)
    if not st.session_state.research:
        icon="📄" if mode=="pdf" else "🌐"
        label=("Upload & process a PDF, then click Run Full Pipeline"
               if mode=="pdf" else "Enter a topic and click One-Click Research")
        st.markdown(f'<div class="rm-empty"><span class="rm-empty-icon">{icon}</span>'
                    f'<div class="rm-empty-title">No report yet</div>'
                    f'<div class="rm-empty-desc">{label}</div></div>',unsafe_allow_html=True)
        return
    for section,content in st.session_state.research.items():
        st.markdown(f'<div class="rm-section"><div class="rm-section-tag">{section}</div>'
                    f'<div class="rm-section-text">{content}</div></div>',unsafe_allow_html=True)
    if st.session_state.summary:
        st.markdown(f'<div class="rm-summary"><div class="rm-summary-label">⚡ Executive Summary</div>'
                    f'<div class="rm-summary-text">{st.session_state.summary}</div></div>',unsafe_allow_html=True)
    sources=st.session_state.web_sources if mode=="general" else []
    pdf_bytes=build_research_pdf(st.session_state.topic,st.session_state.research,
                                  st.session_state.summary,sources,mode=mode)
    st.download_button("⬇️ Download Full Report as PDF",data=pdf_bytes,
        file_name=f"{st.session_state.topic.replace(' ','_')}_Report.pdf",
        mime="application/pdf",use_container_width=True)


# ── NAV ────────────────────────────────────────────────────────────────────────
def render_nav():
    st.markdown("""
    <div class="rm-nav">
      <div class="rm-brand">
        <div class="rm-logo">RM</div>
        <div><div class="rm-title">ResearchMind</div><div class="rm-sub">Multi-Agent Research Assistant</div></div>
      </div>
      <div class="rm-badge">
        <div class="rm-badge-dot"></div>
        <span class="rm-badge-text">All systems operational</span>
      </div>
    </div>
    """,unsafe_allow_html=True)


# ── MODE SWITCHER ───────────────────────────────────────────────────────────────
def render_mode_switcher():
    mode=st.session_state.mode
    st.markdown('<div class="rm-modes">',unsafe_allow_html=True)
    picked=st.radio("mode",["📄 PDF Research","🌐 General Web Research"],
                    index=0 if mode=="pdf" else 1,
                    horizontal=True,label_visibility="collapsed",key="mode_r")
    st.markdown('</div>',unsafe_allow_html=True)
    new=("pdf" if "PDF" in picked else "general")
    if new!=st.session_state.mode:
        st.session_state.mode=new; st.session_state.agent_status={}; st.rerun()


# ── WORKSPACE: PDF ─────────────────────────────────────────────────────────────
def tab_workspace_pdf():
    left,right=st.columns([1,1.85],gap="large")
    with left:
        st.markdown('<div class="rm-card"><div class="rm-card-title">📄 Source document</div>',unsafe_allow_html=True)
        uploaded=st.file_uploader(
            "📂 Click or drag & drop your PDF here",
            type=["pdf"],
            key="pdf_up",
            help="Maximum file size: 50 MB",
        )
        if uploaded:
            st.markdown(f"""
            <div class="rm-file-badge">
              <span class="rm-file-icon">📎</span>
              <div>
                <div class="rm-file-name">{uploaded.name}</div>
                <div class="rm-file-size">{uploaded.size/1024:.1f} KB · PDF</div>
              </div>
            </div>""",unsafe_allow_html=True)

        st.markdown('<div class="rm-card"><div class="rm-card-title">⚙️ Settings</div>',unsafe_allow_html=True)
        topic=st.text_input("Research topic",value=st.session_state.topic,
                            placeholder="e.g. Machine Learning in Healthcare",key="t_pdf")
        if topic!=st.session_state.topic: st.session_state.topic=topic
        top_k=st.slider("Retrieval depth",2,10,st.session_state.top_k,key="k_pdf")
        st.session_state.top_k=top_k
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="rm-card"><div class="rm-card-title">🎯 Actions</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            if st.button("📄 Process PDF",use_container_width=True,key="btn_pp"):
                if not uploaded: st.warning("Upload a PDF first.")
                else:
                    with st.spinner(): process_pdf(uploaded)
                    st.rerun()
        with c2:
            if st.button("📋 Plan",use_container_width=True,key="btn_mp"):
                with st.spinner(): create_plan()
                st.rerun()
        if st.button("✍️ Generate Report",use_container_width=True,key="btn_gr"):
            with st.spinner(): run_research(); st.rerun()
        if st.button("🚀 Run Full Pipeline",use_container_width=True,type="primary",key="btn_fp"):
            run_full(uploaded_file=uploaded); st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        vc=st.session_state.vector_store.ntotal if st.session_state.vector_store else 0
        fname=st.session_state.last_pdf_name or "—"
        st.markdown(f"""
        <div class="rm-metrics">
          <div class="rm-metric"><div class="rm-metric-label">Document</div>
            <div class="rm-metric-val" style="font-size:0.82rem;line-height:1.3;font-weight:700">{fname[:18]}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Characters</div>
            <div class="rm-metric-val">{len(st.session_state.pdf_text):,}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Chunks</div>
            <div class="rm-metric-val">{len(st.session_state.chunks):,}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Vectors</div>
            <div class="rm-metric-val">{vc:,}</div></div>
        </div>""",unsafe_allow_html=True)
        st.markdown('<div class="rm-card"><div class="rm-card-title">🔄 Agent pipeline</div>',unsafe_allow_html=True)
        _pipeline_bar()
        st.markdown('</div>',unsafe_allow_html=True)
        _render_report("pdf")

# ── WORKSPACE: GENERAL ─────────────────────────────────────────────────────────
def tab_workspace_general():
    left,right=st.columns([1,1.85],gap="large")
    with left:
        st.markdown('<div class="rm-card"><div class="rm-card-title">🌐 Research topic</div>',unsafe_allow_html=True)
        topic=st.text_input("Topic",value=st.session_state.topic,
                            placeholder="e.g. Quantum Computing in Cryptography",
                            label_visibility="collapsed",key="t_gen")
        if topic!=st.session_state.topic: st.session_state.topic=topic
        top_k=st.slider("Context depth",2,10,st.session_state.top_k,key="k_gen")
        st.session_state.top_k=top_k
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="rm-card"><div class="rm-card-title">🎯 Actions</div>',unsafe_allow_html=True)
        if st.button("🌐 Search Web",use_container_width=True,key="btn_sw"):
            if not st.session_state.topic: st.warning("Enter a topic first.")
            else:
                with st.spinner("Searching…"): run_web_search(st.session_state.topic)
                st.rerun()
        c1,c2=st.columns(2)
        with c1:
            if st.button("📋 Plan",use_container_width=True,key="btn_pg"):
                with st.spinner(): create_plan(); st.rerun()
        with c2:
            if st.button("✍️ Report",use_container_width=True,key="btn_rg"):
                with st.spinner(): run_research(); st.rerun()
        if st.button("🚀 One-Click Research",use_container_width=True,type="primary",key="btn_ocr"):
            run_full(); st.rerun()
        st.markdown('</div>',unsafe_allow_html=True)

        if st.session_state.web_sources:
            st.markdown('<div class="rm-card"><div class="rm-card-title">🔗 Sources</div>',unsafe_allow_html=True)
            for s in st.session_state.web_sources[:6]:
                t=s['title'][:52]+("…" if len(s['title'])>52 else "")
                u=s['url'][:48]+("…" if len(s['url'])>48 else "")
                st.markdown(f'<div class="rm-src"><div class="rm-src-title">{t}</div>'
                            f'<div class="rm-src-url">{u}</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

    with right:
        ns=len(st.session_state.web_results); nc=len(st.session_state.general_chunks)
        vs=st.session_state.general_vector_store; nv=vs.ntotal if vs else 0
        st.markdown(f"""
        <div class="rm-metrics">
          <div class="rm-metric"><div class="rm-metric-label">Web sources</div><div class="rm-metric-val">{ns}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Chunks</div><div class="rm-metric-val">{nc}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Vectors</div><div class="rm-metric-val">{nv}</div></div>
          <div class="rm-metric"><div class="rm-metric-label">Sections</div><div class="rm-metric-val">{len(st.session_state.research)}</div></div>
        </div>""",unsafe_allow_html=True)
        st.markdown('<div class="rm-card"><div class="rm-card-title">🔄 Agent pipeline</div>',unsafe_allow_html=True)
        _pipeline_bar()
        st.markdown('</div>',unsafe_allow_html=True)
        _render_report("general")


# ── CHAT TAB ───────────────────────────────────────────────────────────────────
def tab_chat():
    mode=st.session_state.mode; ag=get_agents()
    vs=st.session_state.vector_store if mode=="pdf" else st.session_state.general_vector_store
    if not vs:
        hint="Process a PDF first" if mode=="pdf" else "Run a web search first"
        st.markdown(f'<div class="rm-empty"><span class="rm-empty-icon">💬</span>'
                    f'<div class="rm-empty-title">No source loaded</div>'
                    f'<div class="rm-empty-desc">{hint} to enable chat.</div></div>',unsafe_allow_html=True)
        return
    src=(f"📄 {st.session_state.last_pdf_name}" if mode=="pdf" else f"🌐 {st.session_state.topic}")
    st.markdown(f"""
    <div class="rm-chat">
      <div class="rm-chat-head">
        <div class="rm-chat-head-dot"></div>
        <div class="rm-chat-head-label">Chatting with: {src}</div>
      </div>
      <div class="rm-chat-body">""",unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown('<div style="text-align:center;padding:32px;color:var(--muted);font-size:0.82rem;">'
                    'Ask anything about your source material ↓</div>',unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            is_u=msg["role"]=="user"; cls="u" if is_u else "a"
            av="U" if is_u else "AI"; lbl="You" if is_u else "ResearchMind"
            st.markdown(f'<div class="rm-msg {cls}"><div class="rm-av {cls}">{av}</div>'
                        f'<div class="rm-msg-body"><div class="rm-msg-role">{lbl}</div>'
                        f'<div class="rm-msg-text">{msg["content"]}</div></div></div>',unsafe_allow_html=True)
    st.markdown('</div></div>',unsafe_allow_html=True)
    q=st.chat_input("Ask a question about your source…")
    if q:
        st.session_state.chat_history.append({"role":"user","content":q})
        q_emb=ag["embedding"].embed_query(q, vs.vocabulary, vs.idf)
        chunks=ag["retriever"].retrieve(vs,q_emb,top_k=st.session_state.top_k)
        ans=ag["chat"].answer(q,chunks,mode=mode)
        txt=ans.text if ans.ok else "Could not generate a response. Please try again."
        st.session_state.chat_history.append({"role":"assistant","content":txt}); st.rerun()
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat",key="btn_cc"): st.session_state.chat_history=[]; st.rerun()


# ── REFERENCES TAB ─────────────────────────────────────────────────────────────
def tab_references():
    mode=st.session_state.mode
    if mode=="pdf":
        if not st.session_state.pdf_text:
            st.markdown('<div class="rm-empty"><span class="rm-empty-icon">📄</span>'
                        '<div class="rm-empty-title">No PDF loaded</div>'
                        '<div class="rm-empty-desc">Process a PDF to view extracted content.</div></div>',unsafe_allow_html=True)
            return
        t1,t2=st.tabs(["📝 Extracted Text","🧩 Chunks"])
        with t1: st.text_area("text",st.session_state.pdf_text[:8000],height=380,label_visibility="collapsed")
        with t2:
            for c in st.session_state.chunks[:25]:
                with st.expander(f"Chunk {c.id}  —  chars {c.start} to {c.end}"): st.caption(c.text)
    else:
        if not st.session_state.web_results:
            st.markdown('<div class="rm-empty"><span class="rm-empty-icon">🌐</span>'
                        '<div class="rm-empty-title">No web search yet</div>'
                        '<div class="rm-empty-desc">Run General Web Research to see sources.</div></div>',unsafe_allow_html=True)
            return
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
          <div style="font-size:1rem;font-weight:800;color:var(--text);">
            {len(st.session_state.web_results)} Sources Retrieved
          </div>
          <div style="font-size:0.72rem;color:var(--sub);">
            Topic: <strong style="color:var(--text);">{st.session_state.topic}</strong>
          </div>
        </div>""",unsafe_allow_html=True)
        for i,r in enumerate(st.session_state.web_results,1):
            score_pct=int(r.score*100)
            safe_t=r.title.replace("<","&lt;").replace(">","&gt;")
            safe_s=r.content[:300].replace("<","&lt;").replace(">","&gt;")
            url_d=r.url[:72]+("…" if len(r.url)>72 else "")
            st.markdown(f"""
            <div class="rm-ref">
              <div class="rm-ref-num">{i}</div>
              <div style="flex:1;">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:4px;">
                  <div class="rm-ref-title">{safe_t}</div>
                  <div class="rm-ref-score">{score_pct}%</div>
                </div>
                <div class="rm-ref-url">🔗 <a href="{r.url}" target="_blank"
                  style="color:var(--brand);text-decoration:none;">{url_d}</a></div>
                <div class="rm-ref-snip">{safe_s}{"…" if len(r.content)>300 else ""}</div>
              </div>
            </div>""",unsafe_allow_html=True)


# ── ARCHITECTURE TAB ───────────────────────────────────────────────────────────
def tab_architecture():
    c1,c2=st.columns([1,1],gap="large")
    mode=st.session_state.mode
    with c1:
        st.markdown('<div class="rm-card"><div class="rm-card-title">🤖 Agent registry</div>',unsafe_allow_html=True)
        icons={"PDF Reader Agent":"📄","Chunking Agent":"✂️","Embedding Agent":"🔢",
               "Vector Store Agent":"🗄️","Web Search Agent":"🌐","Retriever Agent":"🔍",
               "Planner Agent":"📋","Research Agent":"✍️","Summarizer Agent":"📊","Chat Agent":"💬"}
        for role in AGENT_ROLES:
            ic=icons.get(role["name"],"🤖")
            st.markdown(f"""
            <div class="rm-agent">
              <div class="rm-agent-ic">{ic}</div>
              <div><div class="rm-agent-name">{role['name']}</div>
              <div class="rm-agent-goal">{role['goal']}</div></div>
            </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rm-card"><div class="rm-card-title">🔧 Runtime status</div>',unsafe_allow_html=True)
        from config import GROQ_API_KEY,TAVILY_API_KEY,GROQ_MODEL
        checks=[("Groq LLM",bool(GROQ_API_KEY),GROQ_MODEL),
        ("Tavily Search",bool(TAVILY_API_KEY),"Web search"),
        ("Numpy Vector Store",True,"In-memory dot product"),
        ("TF-IDF Embeddings",True,"custom NumPy")]
        for name,ok,detail in checks:
            c=("var(--green)" if ok else "var(--red)")
            bg=("var(--green-bg)" if ok else "var(--red-bg)")
            bd=("var(--green-bd)" if ok else "#FCA5A5")
            ic=("✓" if ok else "✗ Missing — add to .env")
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bd};border-radius:9px;
                        padding:10px 14px;margin-bottom:6px;
                        display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span style="font-size:0.82rem;font-weight:700;color:var(--text);">{name}</span>
                <span style="font-size:0.7rem;color:var(--sub);margin-left:8px;">{detail}</span>
              </div>
              <span style="color:{c};font-weight:800;">{ic}</span>
            </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="rm-card"><div class="rm-card-title">🗺️ Pipeline flow</div>',unsafe_allow_html=True)
        if mode=="pdf":
            flow="User → PDF Reader → Chunking → TF-IDF Embedding\n→ Vector Store (numpy) → Retriever → Planner (Groq)\n→ Research (Groq) → Summarizer (Groq) → Report PDF"
        else:
            flow="User → Web Search (Tavily) → Chunking → TF-IDF Embedding\n→ Vector Store (numpy) → Retriever → Planner (Groq)\n→ Research (Groq) → Summarizer (Groq) → Report PDF"
        st.markdown(f'<div class="rm-section-text" style="white-space: pre-wrap;">{flow}</div>', unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
_init()
render_nav()

# ── API KEY CHECK ──
from config import GROQ_API_KEY, TAVILY_API_KEY
if not GROQ_API_KEY:
    st.error("⚠️ **GROQ_API_KEY is not set.** Add it in Railway → Variables tab. Get free key at [console.groq.com](https://console.groq.com)", icon="🔑")
if not TAVILY_API_KEY:
    st.warning("⚠️ **TAVILY_API_KEY is not set.** Web Research won't work. Get free key at [app.tavily.com](https://app.tavily.com)", icon="🌐")

render_mode_switcher()

tabs=st.tabs(["🏠 Workspace","💬 Chat","📚 References","🏗️ Architecture"])
with tabs[0]:
    if st.session_state.mode=="pdf": tab_workspace_pdf()
    else: tab_workspace_general()
with tabs[1]: tab_chat()
with tabs[2]: tab_references()
with tabs[3]: tab_architecture()
