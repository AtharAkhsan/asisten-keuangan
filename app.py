import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Asisten Legal Kemenkeu", page_icon="⚖️")

# --- JUDUL & INTRO ---
st.title("⚖️ Asisten Legal & Peraturan")
st.markdown("Membantu pencarian peraturan keuangan dengan cepat.")

# --- SIDEBAR (Untuk API Key) ---
with st.sidebar:
    st.header("Konfigurasi")
    provider = st.selectbox("Pilih Otak AI", ["Google Gemini (Gratis/Berbayar)", "OpenAI (GPT-4)"])
    
    api_key = st.text_input("Masukkan API Key", type="password")
    st.caption("Dapatkan Google AI Key di: aistudio.google.com (Gratis)")
    
    st.divider()
    st.info("Tips: Masukkan kata kunci seperti 'Perjalanan Dinas' atau nomor peraturan.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file ini ada (hasil dari Tahap 1)
    try:
        df = pd.read_csv("clean_legal_data.csv")
        # Pastikan kolom Link tidak kosong (isi hash # jika kosong)
        df['Link'] = df['Link'].fillna('#')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("File 'clean_legal_data.csv' belum ditemukan. Jalankan script extractor dulu.")
    st.stop()

# --- FUNGSI PENCARIAN ---
def search_regulations(query, df, top_k=10):
    # Pencarian keyword sederhana (case insensitive)
    mask = df['Tentang'].str.contains(query, case=False, na=False) | \
           df['Nomor'].str.contains(query, case=False, na=False)
    results = df[mask].head(top_k)
    return results

# --- INTERFACE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat
if prompt := st.chat_input("Tanya tentang aturan (Contoh: Aturan uang makan 2024)"):
    
    # 1. Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Cari Data Relevan di CSV
    search_results = search_regulations(prompt, df)
    
    # Format data untuk dibaca AI
    context_text = ""
    if not search_results.empty:
        context_text = "Data Peraturan yang ditemukan:\n"
        for index, row in search_results.iterrows():
            link_str = f"(Link: {row['Link']})" if row['Link'] != '#' else "(Link tidak tersedia)"
            context_text += f"- {row['Nomor']}: {row['Tentang']} {link_str}\n"
    else:
        context_text = "Tidak ditemukan peraturan yang pas dengan kata kunci tersebut di database."

    # 3. Panggil AI
    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Mohon masukkan API Key di menu sebelah kiri dulu ya.")
            response_text = "Saya perlu API Key untuk berpikir."
        else:
            try:
                # Setup AI Model
                if "Google" in provider:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                else:
                    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

                # Prompt Engineering
                system_prompt = f"""
                Kamu adalah asisten ahli hukum untuk pegawai Kementerian Keuangan.
                Tugasmu: Menjawab pertanyaan user berdasarkan DATA yang diberikan di bawah.
                
                Aturan main:
                1. Jawab dengan sopan dan formal bahasa Indonesia.
                2. JIKA ada data peraturan yang relevan di bawah, SEBUTKAN Nomor, Judul, dan berikan LINK DOWNLOADNYA persis seperti di data.
                3. Jangan mengarang link. Gunakan link yang disediakan di context.
                4. Jika tidak ada di data, katakan jujur bahwa di database internal tidak ditemukan, tapi berikan saran umum.

                DATA DARI DATABASE:
                {context_text}
                """
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                
                with st.spinner("Menganalisa ribuan peraturan..."):
                    response = llm.invoke(messages)
                    response_text = response.content
                    st.markdown(response_text)
            
            except Exception as e:
                response_text = f"Maaf, ada error koneksi ke AI: {e}"
                st.error(response_text)

    # 4. Simpan respon AI
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # 5. Tampilkan Tabel Data (Opsional, biar ayahmu bisa klik langsung)
    if not search_results.empty:
        with st.expander("Lihat Tabel Referensi Asli"):
            st.dataframe(
                search_results[['Nomor', 'Tentang', 'Link']],
                column_config={
                    "Link": st.column_config.LinkColumn("Link Download")
                },
                hide_index=True
            )