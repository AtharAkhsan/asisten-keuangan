import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Asisten Legal Kemenkeu", page_icon="⚖️", layout="wide")

st.title("⚖️ Asisten Legal & Peraturan")
st.markdown("Membantu pencarian peraturan keuangan & analisis dokumen.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan Google API Key", type="password")
    
    st.divider()
    st.header("📂 Upload Dokumen")
    st.info("Upload surat/aturan (PDF/TXT) untuk dianalisa AI.")
    
    # WIDGET UPLOAD FILE
    uploaded_file = st.file_uploader("Pilih file...", type=["pdf", "txt"])
    
    file_content = ""
    if uploaded_file is not None:
        try:
            with st.spinner("Membaca file..."):
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text: file_content += text
                elif uploaded_file.type == "text/plain":
                    file_content = uploaded_file.read().decode("utf-8")
            
            st.success(f"File berhasil dibaca! ({len(file_content)} karakter)")
            with st.expander("Lihat isi file"):
                st.text(file_content[:500] + "...") 
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clean_legal_data.csv")
        df['Link'] = df['Link'].fillna('#')
        df['Search_Text'] = df['Tentang'].astype(str) + " " + df['Nomor'].astype(str) + " " + df['Jenis'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("⚠️ File database tidak ditemukan. Jalankan script extractor dulu!")
    st.stop()

# --- FUNGSI SMART SEARCH ---
def smart_search(query, df, top_k=15):
    keywords = query.lower().split()
    mask = pd.Series([True] * len(df))
    for word in keywords:
        mask = mask & df['Search_Text'].str.contains(word, case=False, na=False)
    results = df[mask]
    if results.empty and len(keywords) > 1:
        mask_or = pd.Series([False] * len(df))
        for word in keywords:
            mask_or = mask_or | df['Search_Text'].str.contains(word, case=False, na=False)
        results = df[mask_or]
    return results.head(top_k)

# --- FUNGSI PEMBERSIH RESPON (FIX UTAMA) ---
def clean_response(content):
    """Mengubah format aneh [{'type': 'text'...}] menjadi string biasa."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Jika formatnya list of dict (multimodal response)
        text_parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Silakan upload dokumen atau tanya tentang aturan."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Contoh: 'Apakah dokumen yang saya upload sesuai dengan PMK ini?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- PERSIAPAN KONTEKS ---
    response_text = ""
    
    # 1. Cari di Database Excel
    search_results = smart_search(prompt, df)
    found_in_db = not search_results.empty
    
    db_context = ""
    if found_in_db:
        db_context = "REFERENSI DARI DATABASE EXCEL:\n"
        for index, row in search_results.iterrows():
            link_str = f"[Download]({row['Link']})" if row['Link'] != '#' else ""
            db_context += f"- {row['Nomor']} tentang {row['Tentang']} | Status: {row.get('Status_Text', '-')} {link_str}\n"
    else:
        db_context = "TIDAK DITEMUKAN DI DATABASE EXCEL."

    # 2. Siapkan Konteks File Upload
    file_context_prompt = ""
    if file_content:
        file_context_prompt = f"""
        USER MENGUPLOAD DOKUMEN BERIKUT:
        ---------------------------------------------------
        {file_content}
        ---------------------------------------------------
        """

    # --- PANGGIL AI ---
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Masukkan API Key dulu.")
        else:
            status_box = st.empty()
            # Urutan model (Flash V2 -> Flash Latest -> Pro)
            models = ["gemini-2.0-flash", "gemini-flash-latest", "gemini-pro"]
            
            success = False
            for model_name in models:
                try:
                    status_box.caption(f"🤖 Berpikir dengan: {model_name}...")
                    
                    if "Google" in provider if 'provider' in locals() else True:
                         llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                    else:
                         llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
                    
                    system_prompt = f"""
                    Kamu adalah Konsultan Hukum Kementerian Keuangan.
                    
                    SUMBER DATA:
                    1. Database Peraturan (Excel): {db_context}
                    2. Dokumen Upload User: {file_context_prompt}
                    
                    INSTRUKSI:
                    - Jawab pertanyaan user dengan mengaitkan Database dan Dokumen (jika ada).
                    - Jika user minta ringkasan file, ringkaslah.
                    - Tetap sopan dan profesional.
                    """
                    
                    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                    
                    response = llm.invoke(messages)
                    
                    # --- BAGIAN PEMBERSIH YANG BARU ---
                    response_text = clean_response(response.content)
                    
                    status_box.empty()
                    st.markdown(response_text)
                    success = True
                    break 
                except Exception as e:
                    # print(f"Error {model_name}: {e}") # Uncomment untuk debug di terminal
                    continue
            
            if not success:
                st.error("Gagal koneksi ke AI. Coba refresh.")
                response_text = "Gagal memproses."

    if response_text and response_text != "Gagal memproses.":
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        if found_in_db:
            with st.expander("📚 Referensi Aturan Terkait"):
                st.dataframe(
                    search_results[['Nomor', 'Tentang', 'Link']],
                    column_config={"Link": st.column_config.LinkColumn("Link")},
                    hide_index=True,
                    use_container_width=True
                )
