#import librerie necessarie
import streamlit as st
import os
import time
from pypdf import PdfReader
import backend

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from opik import configure


configure(
    use_local=False,
    api_key=os.getenv("OPIK_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),  # "alessiamanna"
)


#configurazione pagina 
st.set_page_config(page_title="My Local Notebook", page_icon="📚", layout="wide")



#css custom per impostare il tema
CSS_FILE = "theme.css"
if os.path.exists(CSS_FILE):
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#creo le cartelle necessarie per vector store e documenti sorgente
os.makedirs("vector_store", exist_ok=True)
os.makedirs("source_documents", exist_ok=True)

#utility function, restituisce la lista di notebook disponibili
def get_available_notebooks():
    return [d for d in os.listdir("vector_store") if os.path.isdir(os.path.join("vector_store", d))]

#clean up chat quando si applica un filtro o si cambia notebook
def reset_chat_state():
    st.session_state.history = []
    st.session_state.rag_chain = None
    st.session_state.full_content_for_tools = ""
    st.session_state.summary = ""
    st.session_state.study_guide = ""
    st.session_state.sources_in_notebook = []


#legge e concatena il testo dei pdf
def read_full_content_from_pdfs(pdf_paths: list) -> str:
    full_text = ""
    for file_path in pdf_paths:
        try:
            reader = PdfReader(file_path)
            full_text += "".join(p.extract_text() or "" for p in reader.pages) + "\n\n"
        except Exception:
            continue
    return full_text

#funzione per mostrare i messaggi. Ruolo utente e ruolo assistant, con diverse impostazioni grafiche
def display_message(role, content, avatar_url):
    message_class = "user-message" if role == "user" else "assistant-message"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    if role == "user":
        st.markdown(
            f"""
            <div class="{message_class}">
                <div class="chat-bubble {bubble_class}">{content}</div>
                <img src="{avatar_url}" class="avatar">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="{message_class}">
                <img src="{avatar_url}" class="avatar">
                <div class="stChatMessage {bubble_class}">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

#main ui
st.title("📚NotebookLM")
st.info("Carica i tuoi documenti, crea dei notebook e interagisci con loro.")

# stato della sessione
if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = None
if "history" not in st.session_state:
    st.session_state.history = []

# gestione notebook con sidebar
with st.sidebar:
    st.header("Gestione Notebook")
    notebooks = get_available_notebooks()
    selected_notebook = st.selectbox("Scegli un notebook esistente", notebooks, index=None, placeholder="Seleziona...")
    new_notebook_name = st.text_input("Oppure crea un nuovo notebook", placeholder="Es: 'Ricerca AI'")

    if st.button("Crea Notebook") and new_notebook_name:
        if new_notebook_name in notebooks:
            st.warning("Un notebook con questo nome esiste già.")
        else:
            os.makedirs(os.path.join("vector_store", new_notebook_name), exist_ok=True)
            os.makedirs(os.path.join("source_documents", new_notebook_name), exist_ok=True)
            st.success(f"Notebook '{new_notebook_name}' creato!")
            st.rerun()

    notebook_to_load = new_notebook_name if new_notebook_name and not selected_notebook else selected_notebook
    if notebook_to_load and notebook_to_load != st.session_state.current_notebook:
        st.session_state.current_notebook = notebook_to_load
        reset_chat_state()
        st.rerun()

    st.divider()

    #se c'è un notebook già caricato, si possono aggiungere altri documenti e processarli nuovamente per indicizzarli
    if st.session_state.current_notebook:
        st.subheader(f"Aggiungi a '{st.session_state.current_notebook}'")
        uploaded_files = st.file_uploader("Carica PDF", accept_multiple_files=True, type=['pdf'])
        if st.button("Processa Documenti") and uploaded_files:
            notebook_source_path = os.path.join("source_documents", st.session_state.current_notebook)
            with st.spinner(f"Salvo e indicizzo i documenti..."):
                file_paths_to_process = []
                for up_file in uploaded_files:
                    save_path = os.path.join(notebook_source_path, up_file.name)
                    with open(save_path, "wb") as f:
                        f.write(up_file.getbuffer())
                    file_paths_to_process.append(save_path)
                backend.build_or_update_notebook(file_paths_to_process, st.session_state.current_notebook)
                st.success("Indicizzazione completata!")
                reset_chat_state()
                st.rerun()

#main area
if not st.session_state.current_notebook:
    st.warning("👈 Seleziona o crea un notebook dalla sidebar per iniziare.")
else:
    st.header(f"Interazione con: `{st.session_state.current_notebook}`")
    index_path = os.path.join("vector_store", st.session_state.current_notebook, "index.faiss")

    if not os.path.exists(index_path):
        st.info("Questo notebook è vuoto. Aggiungi dei documenti e clicca su 'Processa Documenti' per iniziare.")
    else:
        # seleziona notebook
        if "sources_in_notebook" not in st.session_state or not st.session_state.sources_in_notebook:
            st.session_state.sources_in_notebook = backend.get_sources_from_notebook(st.session_state.current_notebook)
        
        #scegli se filtrare su documenti specifici
        source_options = ["Tutti i documenti"] + st.session_state.sources_in_notebook
        selected_source = st.selectbox(
            "Filtra la conversazione su un documento specifico:", options=source_options, key="source_selector"
        )
        
        if "last_selected_source" not in st.session_state:
            st.session_state.last_selected_source = selected_source
        
        if st.session_state.last_selected_source != selected_source:
            reset_chat_state()
            st.session_state.last_selected_source = selected_source
            st.rerun()

        if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
            source_filter = selected_source if selected_source != "Tutti i documenti" else None
            with st.spinner(f"Preparo la conversazione per '{selected_source}'..."):
                st.session_state.rag_chain = backend.prepare_rag_chain(
                    st.session_state.current_notebook, source_filter=source_filter
                )

        #strumenti aggiuntivi di analisi e sintesi
        with st.expander("✍️ Strumenti di Analisi & Riepilogo"):
            current_notebook_source_path = os.path.join("source_documents", st.session_state.current_notebook)
            
            st.subheader("Genera un Riassunto Globale")
            if st.button("Crea Riassunto", key="btn_summarize"):
                all_pdfs_in_notebook = [os.path.join(current_notebook_source_path, f) for f in os.listdir(current_notebook_source_path) if f.endswith(".pdf")]
                if not all_pdfs_in_notebook: st.error("Nessun PDF trovato in questo notebook.")
                else:
                    with st.spinner("Leggo e riassumo..."):
                        full_content = read_full_content_from_pdfs(all_pdfs_in_notebook)
                        st.session_state.full_content_for_tools = full_content
                        st.session_state.summary = backend.summarize_text(full_content)

            if "summary" in st.session_state and st.session_state.summary:
                st.markdown(st.session_state.summary)
                if st.button("🔊 Ascolta il riassunto", key="btn_tts"):
                    with st.spinner("Genero audio..."):
                        audio_file = backend.text_to_speech(st.session_state.summary)
                        if audio_file and os.path.exists(audio_file): st.audio(audio_file)
                        else: st.error("Impossibile generare l'audio.")
            st.markdown("---")
            st.subheader("Crea una Guida allo Studio (Q&A)")
            if st.button("Genera Guida allo Studio", key="btn_study_guide"):
                if "full_content_for_tools" not in st.session_state or not st.session_state.full_content_for_tools:
                    all_pdfs_in_notebook = [os.path.join(current_notebook_source_path, f) for f in os.listdir(current_notebook_source_path) if f.endswith(".pdf")]
                    if not all_pdfs_in_notebook: st.error("Nessun PDF trovato.")
                    else:
                        with st.spinner("Leggo i documenti..."):
                             st.session_state.full_content_for_tools = read_full_content_from_pdfs(all_pdfs_in_notebook)
                if st.session_state.get("full_content_for_tools"):
                     with st.spinner("Gemini sta creando la guida..."):
                        st.session_state.study_guide = backend.generate_study_guide(st.session_state.full_content_for_tools)
            if "study_guide" in st.session_state and st.session_state.study_guide:
                st.markdown(st.session_state.study_guide)
        st.divider()

        #chat
        user_img = "https://raw.githubusercontent.com/alessiamanna/AISE_project/refs/heads/main/user-circle.png"
        bot_img = "https://raw.githubusercontent.com/alessiamanna/AISE_project/refs/heads/main/logo_notebook.png"

        #display messaggi
        for message in st.session_state.history:
            avatar = user_img if message["role"] == "user" else bot_img
            display_message(message["role"], message["content"], avatar)

        if prompt := st.chat_input("Fai una domanda..."):
            st.session_state.history.append({"role": "user", "content": prompt})
            display_message("user", prompt, user_img)
            
            with st.spinner("Sto pensando..."):
                answer, sources = backend.generate_answer(prompt, st.session_state.rag_chain)
            
            message_placeholder = st.empty()
            partial_answer = ""
            for char in answer:
                partial_answer += char 
                message_placeholder.markdown(f"""
                    <div class="assistant-message">
                        <img src="{bot_img}" class="avatar">
                        <div class="chat-bubble assistant-bubble">{partial_answer.strip()}</div>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)
            
            st.session_state.history.append({"role": "assistant", "content": answer})
            
            #sorgenti info per explainability
            if sources: 
                with st.expander("Fonti consultate"):
                    for s in sources:
                        st.info(f"**Da: {s['source']}**\n\n> {s['snippet']}")