import re
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Mandiri Branch Learning Assistant",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# CSS – DARK MODE (Mandiri Style)
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F172A;
    color: #E5E7EB;
}
header[data-testid="stHeader"] { visibility: hidden; }
section.main > div { padding-top: 0.8rem; }

.main-title { font-size: 34px; font-weight: 800; color: #E6F2FF; }
.subtitle { font-size: 14px; color: #CBD5E1; }

.badge {
    background: linear-gradient(90deg, #003A8F, #0050B3);
    color: #F8FAFC;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 12px;
}

.hero-wrap{
    background: #020617;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 14px 40px rgba(0,0,0,0.45);
}
.hero-caption{
    padding: 14px;
    border-top: 1px solid #1E293B;
    text-align:center;
    font-weight:800;
}

.cards{
    display:grid;
    grid-template-columns: repeat(3,1fr);
    gap:14px;
    margin-top: 14px;
}
.card{
    background:#020617;
    border-radius:16px;
    padding:14px;
    box-shadow:0 12px 30px rgba(0,0,0,0.4);
}
.icon{
    font-size:22px;
    margin-bottom:6px;
}

section[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
section[data-testid="stChatMessage"][data-testid-user="true"] {
    background-color: #020617;
    border-left: 5px solid #F9B000;
}
section[data-testid="stChatMessage"][data-testid-user="false"] {
    background-color: #020617;
    border-left: 5px solid #38BDF8;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
c1, c2, c3 = st.columns([1.2, 6.5, 1.2])
with c1:
    st.image("assets/mandiri_logo.png", width=120)

with c2:
    st.markdown("""
    <div class="main-title">Branch Learning Assistant</div>
    <div class="subtitle">
        Konsultasi selling skill • Tanya produk Bank Mandiri • Motivasi & people empowerment
    </div>
    <div style="margin-top:10px;">
        <span class="badge">Internal Learning Assistant</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    if st.button("🧹 Clear"):
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# HERO
# =========================================================
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
st.image("assets/hero_banner.png", use_container_width=True)
st.markdown(
    '<div class="hero-caption">Tingkatkan Selling Skill & Product Knowledge Berbasis Pengalaman Nyata</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FEATURE CARDS
# =========================================================
st.markdown("""
<div class="cards">
  <div class="card">
    <div class="icon">🧠</div>
    <b>Konsultasi Selling Skill</b>
    <p>Strategi tarik nasabah, probing, objection handling, hingga closing.</p>
  </div>
  <div class="card">
    <div class="icon">🏦</div>
    <b>Tanya Produk Mandiri</b>
    <p>Fitur, manfaat, syarat, dan rekomendasi produk sesuai kebutuhan nasabah.</p>
  </div>
  <div class="card">
    <div class="icon">🔥</div>
    <b>Motivasi & Empowerment</b>
    <p>Lesson learned dari kisah nyata sales untuk capai target.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MOTIVATION INTENT DETECTOR
# =========================================================
MOTIVATION_PATTERNS = [
    r"\bmotivasi\b", r"\binspirasi\b", r"\bsemangat\b", r"\bdown\b",
    r"\bcapek\b", r"\bburnout\b", r"\bditolak\b", r"\breject\b",
    r"\btarget\b", r"\bclosing\b", r"\bstuck\b"
]

def is_motivation_intent(text):
    return any(re.search(p, text.lower()) for p in MOTIVATION_PATTERNS)


# =========================================================
# LOAD VECTOR DB
# =========================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =========================================================
# LOAD LLM
# =========================================================
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]
)


# =========================================================
# PROMPTS
# =========================================================
rag_prompt = ChatPromptTemplate.from_template("""
Kamu adalah Branch Learning Assistant Bank Mandiri.

Gunakan konteks untuk menjawab secara profesional, tegas,
dan fokus pada *how to sell*.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
""")

motivation_prompt = ChatPromptTemplate.from_template("""
Kamu adalah **Branch Learning Assistant Bank Mandiri**
dalam peran: **Sales Motivation & People Empowerment Coach (Branch Manager Style)**.

MISI UTAMA:
Mengubah kisah inspiratif dalam **Knowledge Base** menjadi
**study case motivasional yang membangkitkan mental bertahan, konsistensi aktivitas,
dan dorongan eksekusi sales**.

==================================================
ATURAN KETAT (WAJIB):
- **DILARANG mengarang kisah, angka, atau detail baru.**
- Semua narasi dan angka HARUS bersumber dari konteks.
- Kisah harus terasa sebagai **study case nyata**, bukan cerita fiktif.
- Jika konteks tidak cukup detail atau tidak relevan:
  katakan secara eksplisit:
  **"Knowledge saya terbatas mengenai kisah yang relevan untuk kondisi ini."**
  lalu lanjutkan dengan:
  → 3 prinsip mental sales
  → 3 aksi hari ini (tanpa kisah baru).
- Jangan menyebut tokoh nyata, brand eksternal, atau kebijakan internal perusahaan.
- Gaya bahasa: **tegas, membina, reflektif, tidak lebay, tidak menggurui.**

==================================================
KONTEKS (Kisah Inspiratif dari Knowledge Base):
{context}

==================================================
PERTANYAAN USER:
{question}

==================================================
OUTPUT (WAJIB ikuti struktur berikut):

A) **Study Case Inspiratif Berbasis Kisah Nyata (150–250 kata)**  
Tulis sebagai narasi reflektif yang mengalir dan membumi.

Narasi WAJIB memuat (ambil dari konteks, jangan menambah):
- Peran/posisi pelaku (jika ada)
- Kondisi awal (target tertinggal, pipeline kosong, banyak penolakan, kelelahan)
- Hambatan utama yang menguji konsistensi
- Aktivitas nyata yang dilakukan, disertai **angka kuantitatif** jika tersedia  
  (contoh: jumlah call, follow-up, appointment, periode waktu)
- Dampak atau perubahan yang tercatat  
  (jika hasil/angka tidak disebutkan, tulis eksplisit)

Fokuskan cerita pada **keputusan untuk tetap menjalankan aktivitas meskipun hasil belum terlihat**.

--------------------------------------------------
B) **Lesson Learned yang Menguatkan Mental Sales (3 poin)**  
Setiap poin HARUS:
- Mengacu langsung ke perilaku dan/atau angka pada kisah
- Menjelaskan *mengapa* tindakan tersebut penting
- Bisa langsung diterapkan oleh sales lain

Gunakan kalimat tegas dan aplikatif.

--------------------------------------------------
C) **Aksi Hari Ini – Target & Eksekusi (Naratif, Tanpa Tabel)**  
Tulis 3 langkah aksi yang:
- Menyebutkan **target kuantitatif** (contoh: jumlah call atau follow-up)
- Menyebutkan **blok waktu eksekusi** (misal: 60–90 menit)
- Menjelaskan fokus tindakan (siapa yang dihubungi, konteksnya apa)

Jika user tidak memberikan angka pencapaian saat ini,
gunakan placeholder seperti: *pencapaian saat ini: __*.

--------------------------------------------------
D) **Kalimat Penutup yang Menggerakkan**
- Maksimal **2 kalimat pendek**
- Tegas, reflektif, berbasis aksi
- Bukan slogan kosong atau motivasi generik

Contoh gaya (JANGAN DIKUTIP, buat kalimat baru berdasarkan knowledge base):
“Target tercapai bukan karena hasil hari ini, tapi karena aktivitas yang tidak dihentikan.”

Mulai jawab sekarang.
""")


# =========================================================
# CHAT STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "booted" not in st.session_state:
    st.session_state.booted = True
    st.session_state.chat_history.append((
        "Halo, saya mulai belajar dari mana?",
        "Mulai dari selling skill dasar atau langsung tanya produk. "
        "Jika butuh motivasi, cukup ketik: *butuh motivasi*."
    ))


# =========================================================
# DISPLAY CHAT
# =========================================================
for u, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.write(a)


# =========================================================
# INPUT
# =========================================================
user_input = st.chat_input("Masukkan pertanyaan anda...")


# =========================================================
# RAG EXECUTION
# =========================================================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    if is_motivation_intent(user_input):
        with st.spinner("Mengambil lesson learned dari kisah inspiratif..."):
            chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | motivation_prompt
                | llm
                | StrOutputParser()
            )
            response = chain.invoke(user_input)
    else:
        with st.spinner("Menganalisis knowledge base..."):
            chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | rag_prompt
                | llm
                | StrOutputParser()
            )
            response = chain.invoke(user_input)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append((user_input, response))
