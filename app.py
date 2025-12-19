import streamlit as st
import pandas as pd
import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Asisten Legal Kemenkeu", page_icon="⚖️", layout="wide")

st.title("⚖️ Asisten Legal & Peraturan")
st.markdown("Membantu pencarian peraturan keuangan & analisis dokumen.")

# --- FILE PENYIMPANAN RIWAYAT ---
HISTORY_FILE = "riwayat_chat.json"

def load_chat_history():
    """Membaca riwayat chat dari file JSON agar tidak hilang saat refresh."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return [] 
    return []

def save_chat_history(messages):
    """Menyimpan riwayat chat ke file JSON."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan Google API Key", type="password")
    
    st.divider()
    st.header("📂 Upload Dokumen")
    st.info("Upload surat/aturan (PDF/TXT) untuk dianalisa AI.")
    
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

    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", type="primary"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.session_state.messages = []
        st.rerun()

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

# --- FUNGSI SMART SEARCH (SUDAH DIPERBAIKI) ---
def smart_search(query, df, top_k=15):
    # Bersihkan query dari karakter aneh agar tidak error split
    clean_query = query.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    keywords = clean_query.lower().split()
    
    # Filter Logika AND
    mask = pd.Series([True] * len(df))
    for word in keywords:
        # FIX: Tambahkan regex=False agar simbol tidak dianggap kode
        mask = mask & df['Search_Text'].str.contains(word, case=False, na=False, regex=False)
    
    results = df[mask]
    
    # Fallback: Logika OR
    if results.empty and len(keywords) > 1:
        mask_or = pd.Series([False] * len(df))
        for word in keywords:
            # FIX: Tambahkan regex=False di sini juga
            mask_or = mask_or | df['Search_Text'].str.contains(word, case=False, na=False, regex=False)
        results = df[mask_or]
        
    return results.head(top_k)

# --- FUNGSI PEMBERSIH RESPON ---
def clean_response(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
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
    saved_history = load_chat_history()
    if saved_history:
        st.session_state.messages = saved_history
    else:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya siap membantu riset dan analisis peraturan."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- PERSIAPAN KONTEKS ---
    response_text = ""
    
    # Gunakan prompt yang asli untuk pencarian data
    search_results = smart_search(prompt, df)
    found_in_db = not search_results.empty
    
    db_context = ""
    if found_in_db:
        db_context = "REFERENSI DARI DATABASE EXCEL (Hanya yang relevan dengan kata kunci):\n"
        for index, row in search_results.iterrows():
            link_str = f"[Download]({row['Link']})" if row['Link'] != '#' else ""
            db_context += f"- {row['Nomor']} tentang {row['Tentang']} | Status: {row.get('Status_Text', '-')} {link_str}\n"
    else:
        db_context = "TIDAK DITEMUKAN DATA SPESIFIK DI DATABASE EXCEL (Gunakan Pengetahuan Umum)."

    file_context_prompt = ""
    if file_content:
        file_context_prompt = f"USER MENGUPLOAD DOKUMEN: {file_content[:15000]}..." # Batasi karakter agar tidak overload

    # --- PANGGIL AI ---
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Masukkan API Key dulu.")
        else:
            status_box = st.empty()
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
                    Kamu adalah Konsultan Ahli Strategi & Hukum Kemenkeu.
                    
                    SUMBER DATA: 
                    1. Database Excel: {db_context}
                    2. Upload User: {file_context_prompt}
                    
                    INSTRUKSI:
                    1. Jawab pertanyaan user dengan SANGAT LENGKAP dan MENDALAM (seperti paper akademis).
                    2. Jika pertanyaan menyangkut konsep baru (seperti Floating Port), gunakan PENGETAHUAN UMUM kamu untuk melakukan benchmarking dan analisis.
                    3. Tetap cek apakah ada aturan di Database Excel yang mungkin berhubungan (misal: aturan Kawasan Pabean, Logistik, atau Investasi), jika ada, sebutkan.
                    4. Struktur jawaban harus rapi (Poin-poin, Analisis, Kesimpulan).
                    """
                    
                    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                    
                    response = llm.invoke(messages)
                    response_text = clean_response(response.content)
                    
                    status_box.empty()
                    st.markdown(response_text)
                    success = True
                    break 
                except Exception as e:
                    # Uncomment baris bawah ini jika ingin melihat error di terminal untuk debugging
                    # print(f"Error {model_name}: {e}") 
                    continue
            
            if not success:
                st.error("Gagal koneksi ke AI.")
                response_text = "Gagal memproses."

    if response_text and response_text != "Gagal memproses.":
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        save_chat_history(st.session_state.messages)
        
        if found_in_db:
            with st.expander("📚 Referensi Aturan Terkait"):
                st.dataframe(
                    search_results[['Nomor', 'Tentang', 'Link']],
                    column_config={"Link": st.column_config.LinkColumn("Link")},
                    hide_index=True,
                    use_container_width=True
                )
