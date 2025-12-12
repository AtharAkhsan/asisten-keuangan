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

# 3. Panggil AI (SISTEM ROTASI MODEL / FALLBACK)
    response_text = ""
    
    # Daftar model urut dari yang tercanggih ke yang paling ringan
    # Kita ambil dari list yang kamu berikan tadi
    model_rotation = [
        "gemini-2.0-flash",           # Prioritas 1: Paling baru & Cepat
        "gemini-flash-latest",        # Prioritas 2: Versi 1.5 Stabil
        "gemini-2.0-flash-lite-preview-02-05", # Prioritas 3: Versi Ringan
        "gemini-pro-latest",          # Prioritas 4: Versi Pro
        "gemini-2.0-flash-exp"        # Prioritas 5: Eksperimental
    ]

    with st.chat_message("assistant"):
        if not api_key:
            st.warning("Mohon masukkan API Key di menu sebelah kiri dulu ya.")
            response_text = "Saya menunggu API Key darimu."
        else:
            # Container untuk status loading
            status_box = st.empty()
            
            # LOGIKA PERULANGAN (LOOP) UNTUK MENCOBA SATU PER SATU
            success = False
            last_error = ""

            for model_name in model_rotation:
                try:
                    status_box.markdown(f"🔄 *Mencoba berpikir menggunakan model: `{model_name}`...*")
                    
                    # 1. Setup Model saat ini
                    if "Google" in provider:
                        llm = ChatGoogleGenerativeAI(
                            model=model_name, 
                            google_api_key=api_key
                        )
                    else:
                        # Jika OpenAI, hanya pakai 1 model (tidak dirotasi di kode ini)
                        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
                        success = True # Bypass loop logic for OpenAI
                        
                    # 2. Siapkan Prompt
                    system_prompt = f"""
                    Kamu adalah asisten ahli hukum Kementerian Keuangan.
                    Tugas: Jawab pertanyaan berdasarkan DATA di bawah.
                    
                    Aturan:
                    1. Jawab sopan & formal.
                    2. WAJIB sertakan Nomor, Judul, dan LINK DOWNLOAD dari data.
                    3. Jika tidak ada di data, katakan tidak ditemukan di database.

                    DATA DATABASE:
                    {context_text}
                    """
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt)
                    ]
                    
                    # 3. Eksekusi
                    response = llm.invoke(messages)
                    response_text = response.content
                    
                    # Jika sampai sini tidak error, berarti SUKSES!
                    success = True
                    status_box.empty() # Hapus status loading
                    st.markdown(response_text)
                    break # KELUAR DARI LOOP KARENA SUDAH BERHASIL
                    
                except Exception as e:
                    # Jika gagal, catat errornya dan LANJUT ke model berikutnya
                    error_code = str(e)
                    print(f"Gagal pakai {model_name}: {error_code}")
                    last_error = error_code
                    continue # Coba model selanjutnya di list
            
            # Jika sudah mencoba SEMUA model dan masih gagal
            if not success:
                st.error(f"⚠️ Maaf, semua model AI sedang sibuk/habis kuota. Coba lagi 1 menit lagi.\nError terakhir: {last_error}")
                response_text = "Gagal memproses permintaan."

    # 4. Simpan respon AI ke History
    if response_text and response_text != "Gagal memproses permintaan.":
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # 5. Tampilkan Tabel Data
    if not search_results.empty:
        with st.expander("Lihat Tabel Referensi Asli"):
            st.dataframe(
                search_results[['Nomor', 'Tentang', 'Link']],
                column_config={"Link": st.column_config.LinkColumn("Link Download")},
                hide_index=True
            )


