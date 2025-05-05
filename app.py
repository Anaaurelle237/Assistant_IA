import os
import chainlit as cl
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
import numpy as np

# Configurer les variables d'environnement
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# Configuration
CONFIG = {
    "markdown_dir": "./markdown_branchements",
    "faiss_dir": "./faiss_index",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_model": "bge-m3:latest",
    "ollama_model": "mistral:7b",
    "top_k_retrieve": 10,
    "top_k_rerank": 3,
    "chunk_size": 512,
    "chunk_overlap": 50
}

# Définir le prompt
PROMPT = PromptTemplate(
    template="""Tu es un assistant expert du processus de branchement chez ENEO. Tes réponses doivent être :
    - Précises et basées exclusivement sur les documents fournis
    - En français courant et facile à comprendre
    - Structurées avec des listes à puces quand c'est pertinent
    - Précise le nom des documents d'où proviennent tes réponses
    Contexte : {context}
    Question : {input}
    Réponds en t'appuyant sur le contexte fourni. Si tu ne sais pas, dis que tu n'as pas l'information.""",
    input_variables=["context", "input"]
)

# Classe RobustTextLoader (copiée depuis votre notebook)
class RobustTextLoader(DirectoryLoader):
    def __init__(self, file_path, encoding="utf-8", fallback_encodings=["iso-8859-1", "cp1252", "utf-16"]):
        super().__init__(file_path, encoding=encoding)
        self.fallback_encodings = fallback_encodings

    def lazy_load(self):
        for encoding in [self.encoding] + self.fallback_encodings:
            try:
                with open(self.file_path, encoding=encoding) as f:
                    text = f.read()
                yield Document(page_content=text, metadata={"source": self.file_path})
                return
            except UnicodeDecodeError:
                continue
            print(f"Échec du chargement de {self.file_path} : impossible de décoder.")

# Fonctions utilitaires (copiées/adaptées depuis votre notebook)
def setup_search_embeddings():
    return HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"], model_kwargs={"device": "cpu"})

def load_documents():
    loader = DirectoryLoader(
        CONFIG["markdown_dir"],
        glob="**/*.md",
        loader_cls=RobustTextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"])
    return splitter.split_documents(documents)

def create_vectorstore(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(CONFIG["faiss_dir"])
    return vectorstore

def load_vectorstore(embedding_model):
    return FAISS.load_local(CONFIG["faiss_dir"], embedding_model, allow_dangerous_deserialization=True)

def setup_reranker():
    return OllamaEmbeddings(model=CONFIG["reranker_model"])

def setup_ollama():
    try:
        return Ollama(model=CONFIG["ollama_model"])
    except Exception as e:
        raise Exception("Erreur lors de la connexion à Ollama. Assurez-vous qu'Ollama est en cours d'exécution.") from e

def create_custom_retriever(vectorstore, reranker_model):
    class CustomRetriever(BaseRetriever):
        vectorstore: FAISS
        reranker_model: OllamaEmbeddings
        top_k_retrieve: int
        top_k_rerank: int

        def _get_relevant_documents(self, query: str) -> list[Document]:
            initial_docs = self.vectorstore.similarity_search(query, k=self.top_k_retrieve)
            query_embedding = np.array(self.reranker_model.embed_query(query))
            doc_embeddings = np.array([self.reranker_model.embed_query(doc.page_content) for doc in initial_docs])
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            scored_docs = list(zip(initial_docs, similarities))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:self.top_k_rerank]]

    return CustomRetriever(
        vectorstore=vectorstore,
        reranker_model=reranker_model,
        top_k_retrieve=CONFIG["top_k_retrieve"],
        top_k_rerank=CONFIG["top_k_rerank"]
    )

def create_chain(llm, retriever):
    combine_docs_chain = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)

# Initialisation à la connexion de Chainlit
@cl.on_chat_start
async def start():
    try:
        embedding_model = setup_search_embeddings()
        vectorstore = (load_vectorstore(embedding_model) if os.path.exists(CONFIG["faiss_dir"])
                      else create_vectorstore(split_documents(load_documents()), embedding_model))
        reranker_model = setup_reranker()
        retriever = create_custom_retriever(vectorstore, reranker_model)
        llm = setup_ollama()
        chain = create_chain(llm, retriever)
        cl.user_session.set("chain", chain)
        await cl.Message(content="Bienvenue ! Posez vos questions sur le processus de branchement ENEO.").send()
    except Exception as e:
        await cl.Message(content=f"Une erreur est survenue : {str(e)}").send()

# Gestion des messages entrants
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    response = await chain.ainvoke({"input": message.content})
    await cl.Message(content=response["answer"]).send()