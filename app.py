import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Asisten Legal Kemenkeu", page_icon="⚖️", layout="wide")

# --- CUSTOM CSS AGAR LEBIH CANTIK ---
st.markdown("""
<style>
    .stChatMessage {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
    .stAlert {border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Asisten Legal & Peraturan")
st.markdown("Membantu pencarian peraturan keuangan dengan analisis AI.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Masukkan Google API Key", type="password")
    
    st.divider()
    st.markdown("**Tips Pencarian:**")
    st.markdown("- Gunakan kata kunci spesifik (misal: *Perjalanan Dinas*, *Gaji 13*).")
    st.markdown("- AI akan mencari di database Excel dulu.")
    st.markdown("- Jika tidak ada, AI akan menggunakan pengetahuan umumnya.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clean_legal_data.csv")
        df['Link'] = df['Link'].fillna('#')
        # Gabungkan semua teks penting ke satu kolom untuk pencarian
        df['Search_Text'] = df['Tentang'].astype(str) + " " + df['Nomor'].astype(str) + " " + df['Jenis'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("⚠️ File database tidak ditemukan. Jalankan script extractor dulu!")
    st.stop()

# --- FUNGSI PENCARIAN PINTAR (SMART SEARCH) ---
def smart_search(query, df, top_k=15):
    # 1. Bersihkan query
    keywords = query.lower().split()
    
    # 2. Filter Logika AND (Semua kata harus ada)
    # Misal: "Uang Makan" -> Baris harus mengandung "uang" DAN "makan"
    mask = pd.Series([True] * len(df))
    for word in keywords:
        mask = mask & df['Search_Text'].str.contains(word, case=False, na=False)
    
    results = df[mask]
    
    # 3. Fallback: Jika Logika AND kosong, pakai Logika OR (Salah satu kata ada)
    if results.empty and len(keywords) > 1:
        mask_or = pd.Series([False] * len(df))
        for word in keywords:
            mask_or = mask_or | df['Search_Text'].str.contains(word, case=False, na=False)
        results = df[mask_or]
        
    return results.head(top_k)

# --- INTERFACE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada aturan apa yang ingin dicari hari ini? Saya siap membantu."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- PROSES PEMIKIRAN AI ---
    response_text = ""
    
    # 1. Cari Data Dulu
    search_results = smart_search(prompt, df)
    found_in_db = not search_results.empty
    
    context_text = ""
    if found_in_db:
        context_text = "REFERENSI DARI DATABASE INTERNAL:\n"
        for index, row in search_results.iterrows():
            link_str = f"[Download]({row['Link']})" if row['Link'] != '#' else "(Link tidak tersedia)"
            context_text += f"- {row['Nomor']} tentang {row['Tentang']} | Status: {row.get('Status_Text', '-')} | {link_str}\n"
    else:
        context_text = "TIDAK DITEMUKAN REFERENSI DI DATABASE EXCEL USER."

    # 2. Panggil AI
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Butuh API Key untuk menjawab.")
        else:
            status_box = st.empty()
            
            # Daftar Model untuk Fallback
            models = ["gemini-2.0-flash", "gemini-flash-latest", "gemini-pro"]
            
            success = False
            for model_name in models:
                try:
                    status_box.caption(f"🤖 Menggunakan otak: {model_name}...")
                    
                    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
                    
                    # --- PROMPT YANG LEBIH CERDAS ---
                    system_prompt = f"""
                    Kamu adalah Konsultan Hukum Senior di Kementerian Keuangan.
                    Tugasmu adalah menjawab pertanyaan user dengan komprehensif.

                    INSTRUKSI KHUSUS:
                    1.  **Analisis Maksud:** Pahami apa yang sebenarnya dicari user (misal: "tukin" = tunjangan kinerja).
                    2.  **Prioritas Database:** Gunakan data di bawah ("REFERENSI DARI DATABASE") sebagai sumber utama.
                        - Jika ada aturan yang pas, jelaskan sedikit isinya dan berikan Link Downloadnya.
                    3.  **Fallback Cerdas:** JIKA di database KOSONG atau kurang relevan:
                        - Gunakan pengetahuan umum kamu tentang hukum Indonesia.
                        - TAPI WAJIB berikan peringatan: "Data spesifik tidak ditemukan di database Excel, namun berdasarkan peraturan umum..."
                    4.  **Gaya Bahasa:** Profesional, membantu, dan terstruktur. Jangan kaku.

                    DATA PENDUKUNG:
                    {context_text}
                    """
                    
                    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                    
                    response = llm.invoke(messages)
                    response_text = response.content
                    
                    status_box.empty()
                    st.markdown(response_text)
                    success = True
                    break # Berhasil, keluar loop
                    
                except Exception:
                    continue # Coba model lain
            
            if not success:
                st.error("Maaf, AI sedang gangguan koneksi. Coba refresh.")
                response_text = "Error koneksi."

    if response_text and response_text != "Error koneksi.":
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Tampilkan Tabel (Hanya jika ada data di DB)
        if found_in_db:
            with st.expander("📚 Lihat Daftar Dokumen Asli (Klik untuk Download)"):
                st.dataframe(
                    search_results[['Nomor', 'Tentang', 'Link']],
                    column_config={"Link": st.column_config.LinkColumn("Link File")},
                    hide_index=True,
                    use_container_width=True
                )
