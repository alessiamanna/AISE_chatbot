'''rag che prende pdf, ne estrae il testo, spezzetta in chunk per la costruzione di embedding che vengono
salvati in un db vettoriale (FAISS). Quando viene fatta una domanda, si recuperano i pezzi più pertinenti dai documenti
e Gemini si occupa di comporre una risposta basandosi solo su quei pezzi. Sono proposte anche utility per riassunit, guide studio e sintesi vocali.
'''

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
from opik import configure

from opik import track


#import langchain e faiss 
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

#gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

#utility per la gestione dei pdf, le variabili d'ambiente, text to speech
from pypdf import PdfReader
from dotenv import load_dotenv
import pyttsx3

#setup .env
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Imposta GOOGLE_API_KEY (o GEMINI_API_KEY) nel .env per usare Gemini.")

#ignore warning asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

#funzioni per gestire document ingestion

#splitta i documenti in chunk con overlap in modo da non perdere il contesto. Ad ogni chunk si dà un metadato soource per consentire di risalire alla fonte
def _split_doc_to_documents(text: str, source: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=ch, metadata={"source": source}) for ch in chunks]


#conversione dei chunk in embedding. Se c'è già indice FAISS lo si carica e lo si unisce ai nuovi documenti se presenti
from opik import track

@track
def build_or_update_notebook(list_of_files: List, notebook_name: str) -> None:

    docs: List[Document] = []
    for file_path in list_of_files:
        try:
            reader = PdfReader(file_path)
            text = "".join(p.extract_text() or "" for p in reader.pages)
            source_name = Path(file_path).name
            docs.extend(_split_doc_to_documents(text, source=source_name))
        except Exception as e:
            print(f"[WARN] Impossibile leggere {file_path}: {e}")

    if not docs:
        print("⚠️ Nessun documento valido da indicizzare.")
        return

    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
    )

    vs_path = os.path.join("vector_store", notebook_name)
    index_file_path = os.path.join(vs_path, "index.faiss")

    if os.path.exists(index_file_path):
        print(f"Aggiornamento del notebook esistente: '{notebook_name}'")
        loaded_db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        new_docs_db = FAISS.from_documents(docs, embeddings)
        loaded_db.merge_from(new_docs_db)
        loaded_db.save_local(vs_path)
        print(f"Notebook '{notebook_name}' aggiornato con successo.")
    else:
        print(f"Creazione di un nuovo indice per il notebook: '{notebook_name}'")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(vs_path)
        print(f"Notebook '{notebook_name}' creato con successo.")



#recupero fonti dall'indice
@track
def get_sources_from_notebook(notebook_name: str) -> List[str]:
    #carico indice faiss
    vs_path = os.path.join("vector_store", notebook_name)
    if not os.path.exists(vs_path):
        return []

    try:
        _ensure_event_loop()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        
        all_metadata = [doc.metadata for doc in db.docstore._dict.values()] #estrae i metadata
        unique_sources = sorted(list(set(meta['source'] for meta in all_metadata if 'source' in meta)))
        return unique_sources
    except Exception as e:
        print(f"[get_sources_from_notebook] Errore: {e}")
        return []

#prompt generico dato che non è un chatbot specializzato. Specifico che non deve divagare, usare conoscenza esterna e deve offire spiegazioni chiare e comprensibili,
#in quanto chatbot pensato per il supporto allo studio
NOTEBOOK_SYSTEM_PROMPT = (
    "Sei un assistente intelligente che aiuta a esplorare e comprendere un set di documenti. "
    "Rispondi alle domande basandoti ESCLUSIVAMENTE sul contesto fornito. "
    "Sii preciso, cita le fonti se possibile e se la risposta non è nel contesto, dichiara di non avere l'informazione. "
    "Non usare conoscenze esterne. Non inventare risposte. Spiega il contenuto dei file che ti viene richiesto in maniera chiara e comprensibile.\n\n"
    "Contesto fornito: {context}"
)

#rag chain
@track
def prepare_rag_chain(
    notebook_name: str,
    temperature: float = 0.2,
    max_length: int = 2048,
    source_filter: str = None
):

    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    vs_path = os.path.join("vector_store", notebook_name)
    if not os.path.exists(vs_path):
        raise FileNotFoundError(f"Il notebook '{notebook_name}' non esiste.")

    loaded_db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    #restituisce 4 chunk simili
    search_kwargs = {"k": 4}
    if source_filter: #se source filter attivo, recupera solo dagli elementi selezionati
        search_kwargs["filter"] = {"source": source_filter}
        print(f"Retriever attivato con filtro: source='{source_filter}'")

    retriever = loaded_db.as_retriever(search_kwargs=search_kwargs)

    system_message = SystemMessagePromptTemplate.from_template(NOTEBOOK_SYSTEM_PROMPT)
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    qa_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    #inizializza LLM 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=temperature, max_output_tokens=max_length, google_api_key=GOOGLE_API_KEY
    )

    #memoria conversazionale con ultimi 5 msg
    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True, output_key="answer"
    )

    #conversationalretrievalchain, quando si fa una domanda al chatbot la chain fa:
    #1. cerca chunk più rilevanti
    #2. li passa a gemini
    #3. ottiene una risposta coerente
    #  
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

#generazione risposta, invoca la catena con la domanda
@track
def generate_answer(question: str, rag_chain) -> Tuple[str, List[dict]]:
    try:
        response = rag_chain.invoke({"question": question})
        answer = (response.get("answer") or "").strip()
        docs = response.get("source_documents", []) or []
        
        sources = []
        for d in docs:
            source_name = d.metadata.get("source", "N/D")
            snippet = d.page_content.strip()[:200] + "..." 
            sources.append({"source": source_name, "snippet": snippet})
            
        return answer, sources
    except Exception as e:
        print(f"[generate_answer] Errore: {e}")
        return "Si è verificato un problema nel generare la risposta.", []

#llm per sintesi in punti chiave dei documenti forniti.
@track
def summarize_text(full_text: str) -> str:
    if not full_text: return "Nessun testo da riassumere."
    _ensure_event_loop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un esperto nel sintetizzare documenti complessi. Crea un riassunto dettagliato del testo, organizzando i concetti principali in punti chiave."),
        ("human", "Testo da riassumere:\n\n{text_content}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text_content": full_text})
        return response.content
    except Exception as e:
        print(f"[summarize_text] Errore: {e}")
        return "Impossibile generare il riassunto."

#llm per fornire guida allo studio con domande e relative risposte
@track
def generate_study_guide(full_text: str) -> str:
    if not full_text: return "Nessun testo su cui generare una guida."
    _ensure_event_loop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un assistente allo studio. Analizza il testo e crea una guida (domande e risposte brevi) sui concetti chiave. Formatta in Markdown:\n**Domanda 1:** ...\n**Risposta:** ..."),
        ("human", "Testo da cui creare la guida:\n\n{text_content}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text_content": full_text})
        return response.content
    except Exception as e:
        print(f"[generate_study_guide] Errore: {e}")
        return "Impossibile generare la guida allo studio."

#pyttsx3 per fare conversione del testo TTS, riproducibile direttamente tramite streamlit
@track
def text_to_speech(text: str, audio_path: str = "summary_audio.mp3") -> str:
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        return audio_path
    except Exception as e:
        print(f"[text_to_speech] Errore: {e}")
        return ""