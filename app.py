# chainlit_app_optimized.py
import os
import pickle
import logging
import asyncio
from typing import List

import chainlit as cl
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_eneo")

# ---------- Config ----------
CONFIG = {
    "markdown_dir": "./markdown_branchements",
    "source_dir": "./processus_branchement",  #  fichiers sources vers les fichiers source
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "ollama_model": "mistral:7b",
    "top_k_retrieve": 6,   # nombre de chunks récupérés
    "chunk_size": 800,     # taille de chunk
    "chunk_overlap": 150,
    "faiss_index_file": "./faiss_index.pkl",  #base vectorielle
    "device": "cpu"
}

# ---------- Prompt ----------
COMBINE_PROMPT = PromptTemplate(
    template="""
Tu es un assistant expert chez ENEO Cameroun spécialisé dans les procédures de branchement électrique.

INSTRUCTIONS :
- RÉPONDS UNIQUEMENT AVEC L'INFORMATION PRÉSENTE DANS LES DOCUMENTS.
- SI L'INFO N'EST PAS DANS LES DOCUMENTS : "Cette information n'est pas disponible dans mes documents"
- CITE LES SOURCES ENTRE CROCHETS [Document: nom_du_fichier] AVEC UN LIEN CLIQUABLE VERS LE FICHIER SOURCE
- INDIQUE LES DATES DES DOCUMENTS SI DISPONIBLES

DOCUMENTS :
{context}

QUESTION :
{input}

RÉPONSE :
""",
    input_variables=["context", "input"]
)


class RobustTextLoader(TextLoader):
    """Lecture d'un fichier texte en essayant plusieurs encodages (UTF-8 par défaut)."""
    def __init__(self, file_path, encoding="utf-8", fallback_encodings=None):
        self.file_path = file_path
        self.encoding = encoding
        self.fallback_encodings = fallback_encodings or ["utf-8", "iso-8859-1", "cp1252", "utf-16", "latin1", "ascii"]

    def load(self) -> List[Document]:
        for enc in self.fallback_encodings:
            try:
                with open(self.file_path, encoding=enc, errors="strict") as f:
                    text = f.read()
                filename = os.path.basename(self.file_path)
                return [Document(page_content=text, metadata={"source": filename, "full_path": self.file_path, "file_size": len(text)})]
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read {self.file_path} with {enc}: {e}")
                continue
        logger.warning(f"All encodings failed for {self.file_path}")
        return [Document(page_content="", metadata={"source": self.file_path})]

#  fonction pour Générer lien vers fichers source 
def generate_source_links_file():
    """Génère un fichier Markdown avec les liens vers les fichiers sources dans processus_branchement et ses sous-dossiers."""
    source_dir = CONFIG["source_dir"]
    links_file = CONFIG["source_links_file"]
    if not os.path.exists(source_dir):
        logger.warning(f"Source directory {source_dir} not found")
        return []

    source_files = []
    supported_extensions = [".pdf", ".doc", ".docx", ".txt"]  # extensions possibles pour les fichiers sources
    # Scanner récursivement les sous-dossiers
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                source_files.append(os.path.join(root, file))
    
    markdown_files = [f for f in os.listdir(CONFIG["markdown_dir"]) if f.endswith(".md")]
    links = []
    for md_file in markdown_files:
        base_name = os.path.splitext(md_file)[0]
        for source_file in source_files:
            if os.path.splitext(os.path.basename(source_file))[0] == base_name:
                links.append(f"- [{md_file}]( file://{os.path.abspath(source_file)} )")
                break
        else:
            logger.warning(f"No source file found for markdown {md_file}")

    # Écrire les liens des fichiers source dans un fichier Markdown
    try:
        with open(links_file, "w", encoding="utf-8") as f:
            f.write("# Documents Sources\n\n")
            if links:
                f.write("Les documents sources suivants sont disponibles :\n\n")
                f.write("\n".join(links))
            else:
                f.write("Aucun document source trouvé dans le dossier processus_branchement ou ses sous-dossiers.")
        logger.info(f"Source links file generated: {links_file}")
    except Exception as e:
        logger.error(f"Failed to write source links file: {e}")
    
    return links

# Embeddings 
def setup_embeddings():
    logger.info("Initializing HuggingFace embeddings (CPU)...")
    emb = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": CONFIG["device"], "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    return emb

# charger les documents
def load_documents():
    folder = CONFIG["markdown_dir"]
    if not os.path.exists(folder):
        logger.error(f"Folder {folder} not found")
        return []
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=RobustTextLoader, show_progress=False)
    docs = loader.load()
    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    logger.info(f"Loaded {len(docs)} non-empty markdown documents")
    return docs

# découper les documents
def split_documents(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    
    for i, c in enumerate(chunks):
        c.metadata.setdefault("chunk_id", i)
        c.metadata.setdefault("chunk_length", len(c.page_content))
        # Ajouter le lien vers le fichier source dans les métadonnées
        base_name = os.path.splitext(c.metadata["source"])[0]
        source_dir = CONFIG["source_dir"]
        for root, _, files in os.walk(source_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_name and any(file.lower().endswith(ext) for ext in [".pdf", ".doc", ".docx", ".txt"]):
                    source_path = os.path.join(root, file)
                    c.metadata["source_link"] = f"file://{os.path.abspath(source_path)}"
                    break
            else:
                continue
            break
        else:
            c.metadata["source_link"] = None
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


# base vectorielle FAISS

# construction de base vectorielle
def build_vectorstore(chunks, embeddings):
    if not chunks:
        logger.error("No chunks to index")
        return None
    logger.info("Building FAISS index...")
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    try:
        with open(CONFIG["faiss_index_file"], "wb") as f:
            pickle.dump(vs, f)
        logger.info("FAISS index saved.")
    except Exception as e:
        logger.warning(f"Could not save FAISS index: {e}")
    return vs

# chargement de base vectorielle
def load_vectorstore(embeddings):
    if not os.path.exists(CONFIG["faiss_index_file"]):
        return None
    try:
        with open(CONFIG["faiss_index_file"], "rb") as f:
            vs = pickle.load(f)
        # embedding function 
        vs._embedding_function = embeddings
        logger.info("FAISS index loaded from disk.")
        return vs
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None

#  Construction RAG chain 
def create_rag_chain(llm, retriever: BaseRetriever):
    #  stuff chain
    stuff = create_stuff_documents_chain(llm=llm, prompt=COMBINE_PROMPT)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff)
    return chain

# memoire cache
response_cache = {}

# Chainlit 

@cl.on_chat_start
async def start():
    """Initialisation (s'exécute au début de chaque session)."""
    await cl.Message(content="🔄 Initialisation RAG ENEO (optimisée CPU)...").send()

    # Générer le fichier de liens sources
    source_links = generate_source_links_file()
    links_file = CONFIG["source_links_file"]
    if source_links:
        await cl.Message(content=f"📄 La liste des documents sources est disponible dans [source_links.md](file://{os.path.abspath(links_file)}). Vous pouvez aussi taper 'list sources' pour l'afficher.").send()
    else:
        await cl.Message(content="⚠️ Aucun fichier source trouvé dans le dossier processus_branchement ou ses sous-dossiers.").send()

    # embeddings
    embeddings = setup_embeddings()

    # Vérifier d'abord si l'index FAISS existe
    vs = load_vectorstore(embeddings)
    if vs is None:
        # Si l'index n'existe pas, charger les documents et construire l'index
        await cl.Message(content="📚 Aucune base vectorielle trouvée. Lecture des documents et indexation (cela peut prendre quelques dizaines de secondes)...").send()
        docs = load_documents()
        if not docs:
            await cl.Message(content="❌ Aucun document trouvé dans le dossier markdown. Veuillez vérifier.").send()
            return
        chunks = split_documents(docs)
        vs = build_vectorstore(chunks, embeddings)
        if vs is None:
            await cl.Message(content="❌ Échec création du vectorstore.").send()
            return
    else:
        await cl.Message(content="✅ Base vectorielle FAISS chargée avec succès.").send()

    # Configuration du  LLM en streaming
    try:
        llm_main = OllamaLLM(
            model=CONFIG["ollama_model"],
            temperature=0.0,
            top_p=1.0,
            streaming=True  
        )
    except TypeError:
        # fallback si streaming non supporté 
        llm_main = OllamaLLM(model=CONFIG["ollama_model"], temperature=0.0, top_p=1.0)

    # Base retriever (I'll try MultiQueryRetriever pour les reformulations soon)
    retriever = vs.as_retriever(search_kwargs={"k": CONFIG["top_k_retrieve"]})

    # Chain with the retriever
    chain = create_rag_chain(llm_main, retriever)

    # sauvegarde dans une session
    cl.user_session.set("chain", chain)
    cl.user_session.set("vectorstore", vs)
    cl.user_session.set("retriever", retriever)

    await cl.Message(content="✅ Système initialisé. Posez votre question ou tapez 'list sources' pour voir les documents sources.").send()

@cl.on_message
async def main(message: cl.Message):
    """Gestion d'un message utilisateur (optimisée)."""
    q = message.content.strip()
    if not q:
        await cl.Message(content="Veuillez poser une question.").send()
        return

   
    if q.lower() == "clear cache":
        response_cache.clear()
        await cl.Message(content="🗑️ Cache vidé.").send()
        return
    elif q.lower() == "list sources":
        links_file = CONFIG["source_links_file"]
        if os.path.exists(links_file):
            try:
                with open(links_file, "r", encoding="utf-8") as f:
                    content = f.read()
                await cl.Message(content=content).send()
            except Exception as e:
                await cl.Message(content=f"❌ Erreur lors de la lecture de {links_file}: {str(e)[:200]}").send()
        else:
            await cl.Message(content="⚠️ Fichier source_links.md non trouvé. Relancez l'initialisation.").send()
        return

    # renvoi immédiat si réponse en cache 
    if q in response_cache:
        await cl.Message(content=response_cache[q]).send()
        return

    # cherche la chaîne
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="❌ Système non initialisé. Redémarrez la session.").send()
        return

    # message "en cours" envoyé au client (immédiat)
    msg = cl.Message(content="")
    await msg.send()

    # Streaming via chain.astream
    logger.info(f"Start streaming response for query: {q}")
    full_response = ""
    last_chunk = None

    try:
        async for chunk in chain.astream({"input": q}):
            # chunk peut être un dict; on essaye d'extraire tous les champs potentiels
            token_piece = ""
            if isinstance(chunk, str):
                token_piece = chunk
            elif isinstance(chunk, dict):
                token_piece = chunk.get("answer") or chunk.get("result") or chunk.get("response") or ""
                last_chunk = chunk
            else:
                token_piece = str(chunk)

            if token_piece:
                # stream token piece as soon as it arrives (no artificial sleep)
                await msg.stream_token(token_piece)
                full_response += token_piece

        # si la chaîne fournit les documents sources à la fin, on les ajoute à l'affichage avec liens
        if isinstance(last_chunk, dict) and "source_documents" in last_chunk and last_chunk["source_documents"]:
            sources = []
            for d in last_chunk["source_documents"]:
                source_name = d.metadata.get("source", "Unknown")
                source_link = d.metadata.get("source_link", None)
                if source_link:
                    sources.append(f"[{source_name}]({source_link})")
                else:
                    sources.append(source_name)
            sources = list(set(sources))  # éviter les doublons
            sources_text = "\n\n📚 Sources : " + ", ".join(sources)
            await msg.stream_token(sources_text)
            full_response += sources_text

        # cache la réponse complète
        response_cache[q] = full_response
        logger.info("Response cached for query.")
    except Exception as e:
        logger.exception("Error during streaming/generation")
        # en cas d'erreur, on envoie un message d'erreur lisible
        await cl.Message(content=f"❌ Une erreur est survenue pendant la génération : {str(e)[:200]}").send()
        

#lien pour lancer l'app:  chainlit run app.py -w --host 0.0.0.0 --port 8000
#URL: http://10.241.132.49:8000/

    