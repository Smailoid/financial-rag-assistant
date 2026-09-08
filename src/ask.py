import os
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


VECTORSTORE_DIR = Path("vectorstore/faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:3b"


PROMPT_TEMPLATE = """
Tu es un assistant spécialisé en analyse documentaire financière et ESG.

Tu dois répondre uniquement à partir du contexte fourni.
Si le contexte ne contient pas l'information, dis clairement :
"Je ne trouve pas cette information dans les documents fournis."

Réponds en français correct, de manière structurée, concise et professionnelle.
N'invente aucune information.
Privilégie les informations les plus directement liées à la question.

Contexte :
{context}

Question :
{question}

Réponse :
"""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def retrieve_documents(vectorstore, question: str, k: int = 8):
    results = vectorstore.similarity_search_with_score(
        query=question,
        k=k,
    )

    return results


def format_context(results):
    formatted_chunks = []

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "page inconnue")
        display_page = page + 1 if isinstance(page, int) else page

        chunk_text = (
            f"[Source {i}] Fichier : {source}, page : {display_page}, score : {score:.4f}\n"
            f"{doc.page_content}"
        )

        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)


def generate_answer(question: str, results):
    context = format_context(results)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content


def display_sources(results):
    print("\nSources utilisées :")

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "page inconnue")
        display_page = page + 1 if isinstance(page, int) else page

        print(f"{i}. {source} — page {display_page} — score {score:.4f}")


def main():
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "La base vectorielle n'existe pas encore. "
            "Lance d'abord : python src/ingest.py"
        )

    print("Chargement de la base vectorielle...")
    vectorstore = load_vectorstore()

    print("Assistant RAG local prêt.")
    print("Tape 'exit' pour quitter.\n")

    while True:
        question = input("Ta question : ")

        if not question.strip():
            print("Question vide, écris une vraie question.")
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("Fin du programme.")
            break

        results = retrieve_documents(vectorstore, question)
        answer = generate_answer(question, results)

        print("\nRéponse :")
        print(answer)

        display_sources(results)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()