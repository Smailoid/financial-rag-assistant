import os
import shutil
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

load_dotenv()


DATA_DIR = Path("data/reports")
VECTORSTORE_DIR = Path("vectorstore/faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def get_pdf_files(data_dir: Path) -> list[Path]:
    """
    Récupère tous les fichiers PDF présents dans data/reports.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Le dossier {data_dir} n'existe pas. "
            "Crée le dossier data/reports et ajoute au moins un PDF."
        )

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {data_dir}. "
            "Ajoute au moins un fichier PDF dans data/reports."
        )

    return pdf_files


def load_single_pdf(pdf_path: Path):
    """
    Charge un seul PDF avec PyPDFLoader.
    Retourne les pages non vides et quelques statistiques.
    """
    print(f"\nChargement du fichier : {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    total_pages = len(pages)

    non_empty_pages = [
        page for page in pages
        if page.page_content and page.page_content.strip()
    ]

    empty_pages = total_pages - len(non_empty_pages)

    print(f"  Pages détectées       : {total_pages}")
    print(f"  Pages exploitables    : {len(non_empty_pages)}")
    print(f"  Pages vides/illisibles: {empty_pages}")

    if not non_empty_pages:
        print(
            "  ⚠️ Aucun texte exploitable trouvé. "
            "Ce PDF est probablement scanné ou aplati en image."
        )

    return non_empty_pages


def load_all_pdfs(data_dir: Path):
    """
    Charge tous les PDF textuels du dossier.
    Ignore les PDF sans texte exploitable.
    """
    pdf_files = get_pdf_files(data_dir)

    all_documents = []
    skipped_files = []

    print("=" * 80)
    print("Début de l'ingestion des documents")
    print("=" * 80)
    print(f"Dossier source : {data_dir}")
    print(f"Nombre de PDF trouvés : {len(pdf_files)}")

    for pdf_path in pdf_files:
        try:
            documents = load_single_pdf(pdf_path)

            if documents:
                all_documents.extend(documents)
            else:
                skipped_files.append(pdf_path.name)

        except Exception as error:
            print(f"  ❌ Erreur pendant la lecture de {pdf_path.name} : {error}")
            skipped_files.append(pdf_path.name)

    if not all_documents:
        raise ValueError(
            "Aucun document exploitable n'a été chargé. "
            "Vérifie que tes PDF contiennent du texte sélectionnable."
        )

    print("\n" + "=" * 80)
    print("Résumé chargement")
    print("=" * 80)
    print(f"Pages exploitables totales : {len(all_documents)}")

    if skipped_files:
        print("PDF ignorés :")
        for file_name in skipped_files:
            print(f"  - {file_name}")
    else:
        print("Aucun PDF ignoré.")

    return all_documents


def split_documents(documents):
    """
    Découpe les pages en chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "Les documents ont été chargés, mais aucun chunk n'a été créé."
        )

    print("\n" + "=" * 80)
    print("Découpage en chunks")
    print("=" * 80)
    print(f"Nombre de chunks créés : {len(chunks)}")
    print(f"Chunk size              : {CHUNK_SIZE}")
    print(f"Chunk overlap           : {CHUNK_OVERLAP}")

    return chunks


def reset_vectorstore(vectorstore_dir: Path):
    """
    Supprime l'ancien index FAISS avant d'en créer un nouveau.
    """
    if vectorstore_dir.exists():
        print("\nSuppression de l'ancien index FAISS...")
        shutil.rmtree(vectorstore_dir)


def create_vectorstore(chunks):
    """
    Crée les embeddings locaux et sauvegarde la base FAISS.
    """
    print("\n" + "=" * 80)
    print("Création de la base vectorielle")
    print("=" * 80)
    print(f"Modèle d'embedding : {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Base vectorielle sauvegardée dans : {VECTORSTORE_DIR}")

    return vectorstore


def main():
    documents = load_all_pdfs(DATA_DIR)
    chunks = split_documents(documents)

    reset_vectorstore(VECTORSTORE_DIR)
    create_vectorstore(chunks)

    print("\n" + "=" * 80)
    print("Ingestion terminée avec succès")
    print("=" * 80)


if __name__ == "__main__":
    main()