import os
import sys
import warnings
from pathlib import Path

import streamlit as st


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


# Permet d'importer les fonctions depuis src/
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from ask import (  # noqa: E402
    VECTORSTORE_DIR,
    load_vectorstore,
    retrieve_documents,
    generate_answer,
)

from ingest import (  # noqa: E402
    DATA_DIR,
    main as run_ingestion,
)


st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def cached_load_vectorstore():
    """
    Charge la base vectorielle FAISS une seule fois.
    Streamlit réutilise ensuite l'objet en cache.
    """
    return load_vectorstore()


def save_uploaded_files(uploaded_files, replace_existing: bool = False):
    """
    Sauvegarde les fichiers PDF uploadés dans data/reports/.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if replace_existing:
        for existing_pdf in DATA_DIR.glob("*.pdf"):
            existing_pdf.unlink()

    saved_files = []

    for uploaded_file in uploaded_files:
        safe_file_name = Path(uploaded_file.name).name
        destination = DATA_DIR / safe_file_name

        with open(destination, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_files.append(destination.name)

    return saved_files


def format_sources(results):
    sources = []

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "page inconnue")

        display_page = page + 1 if isinstance(page, int) else page
        file_name = Path(source).name

        sources.append(
            {
                "rank": i,
                "file_name": file_name,
                "source": source,
                "page": display_page,
                "score": score,
                "content": doc.page_content,
            }
        )

    return sources


def display_current_pdfs():
    """
    Affiche les PDF actuellement présents dans data/reports.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        st.info("Aucun PDF actuellement présent dans `data/reports/`.")
    else:
        st.markdown("**PDF actuellement disponibles :**")
        for pdf_file in pdf_files:
            st.write(f"- `{pdf_file.name}`")


def main():
    st.title("📊 Financial RAG Assistant")
    st.caption(
        "Assistant local pour interroger des documents financiers / ESG "
        "avec FAISS, embeddings Hugging Face et Ollama."
    )

    with st.sidebar:
        st.header("📄 Gestion des documents")

        uploaded_files = st.file_uploader(
            "Uploader un ou plusieurs PDF",
            type=["pdf"],
            accept_multiple_files=True,
        )

        replace_existing = st.checkbox(
            "Remplacer les PDF existants",
            value=False,
            help=(
                "Si coché, les anciens PDF dans data/reports seront supprimés "
                "avant d'ajouter les nouveaux fichiers."
            ),
        )

        if st.button("Sauvegarder les PDF"):
            if not uploaded_files:
                st.warning("Aucun fichier sélectionné.")
            else:
                saved_files = save_uploaded_files(
                    uploaded_files=uploaded_files,
                    replace_existing=replace_existing,
                )

                st.success("PDF sauvegardés :")
                for file_name in saved_files:
                    st.write(f"- `{file_name}`")

        st.markdown("---")
        display_current_pdfs()

        st.markdown("---")

        if st.button("Reconstruire la base FAISS", type="primary"):
            try:
                with st.spinner("Ingestion des documents en cours..."):
                    run_ingestion()

                cached_load_vectorstore.clear()
                st.success("Base vectorielle reconstruite avec succès.")

            except Exception as error:
                st.error(f"Erreur pendant l'ingestion : {error}")

        st.markdown("---")
        st.header("⚙️ Paramètres")

        k = st.slider(
            "Nombre de passages récupérés",
            min_value=3,
            max_value=12,
            value=8,
            step=1,
        )

        show_chunks = st.checkbox(
            "Afficher le contenu des passages sources",
            value=False,
        )

        st.markdown("---")
        st.markdown("### Pipeline")
        st.markdown(
            """
            1. Upload ou sélection des PDF  
            2. Ingestion des documents  
            3. Création des embeddings  
            4. Recherche FAISS  
            5. Génération locale avec Ollama  
            6. Réponse sourcée  
            """
        )

    if not VECTORSTORE_DIR.exists():
        st.warning(
            "La base vectorielle n'existe pas encore. "
            "Ajoute des PDF puis clique sur **Reconstruire la base FAISS**."
        )
        st.stop()

    st.subheader("Pose une question sur les documents")

    question = st.text_input(
        "Question",
        placeholder="Exemple : Que disent les documents sur la relation entre ESG et performance financière ?",
    )

    ask_button = st.button("Interroger les documents", type="primary")

    if ask_button:
        if not question.strip():
            st.warning("Écris une question avant de lancer la recherche.")
            st.stop()

        with st.spinner("Chargement de la base vectorielle..."):
            vectorstore = cached_load_vectorstore()

        with st.spinner("Recherche des passages pertinents..."):
            results = retrieve_documents(
                vectorstore=vectorstore,
                question=question,
                k=k,
            )

        with st.spinner("Génération de la réponse avec Ollama..."):
            answer = generate_answer(
                question=question,
                results=results,
            )

        st.markdown("## Réponse")
        st.write(answer)

        st.markdown("## Sources utilisées")

        sources = format_sources(results)

        for source in sources:
            with st.expander(
                f"Source {source['rank']} — {source['file_name']} "
                f"— page {source['page']} — score {source['score']:.4f}"
            ):
                st.markdown(f"**Fichier :** `{source['file_name']}`")
                st.markdown(f"**Chemin :** `{source['source']}`")
                st.markdown(f"**Page :** {source['page']}")
                st.markdown(f"**Score FAISS :** `{source['score']:.4f}`")

                if show_chunks:
                    st.markdown("**Passage récupéré :**")
                    st.write(source["content"])

    else:
        st.info(
            "Entre une question, puis clique sur **Interroger les documents**."
        )


if __name__ == "__main__":
    main()