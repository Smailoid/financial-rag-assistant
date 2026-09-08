# Financial RAG Assistant

Assistant RAG local permettant d'interroger des documents financiers et ESG au format PDF.

Ce projet permet de charger des documents PDF, de construire un index vectoriel local, de retrouver les passages les plus pertinents pour une question donnée, puis de générer une réponse sourcée via une interface Streamlit.

## Objectif du projet

L'objectif est de développer un prototype d'assistant documentaire spécialisé dans l'analyse de documents financiers, ESG ou extra-financiers.

Le système repose sur une architecture RAG, c'est-à-dire Retrieval-Augmented Generation :

```text
Documents PDF
→ extraction du texte
→ découpage en chunks
→ embeddings Hugging Face MiniLM
→ index vectoriel FAISS
→ recherche sémantique
→ génération locale avec Llama 3.2 3B via Ollama
→ réponse sourcée dans Streamlit
```

## Architecture finale

La version actuelle est entièrement locale et utilise :

- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face
- **Base vectorielle** : FAISS
- **Modèle de génération** : Llama 3.2 3B via Ollama
- **Interface utilisateur** : Streamlit
- **Chargement PDF** : PyPDF / LangChain
- **Découpage du texte** : `RecursiveCharacterTextSplitter`

Aucune clé API OpenAI n'est nécessaire pour exécuter la version actuelle.

## Fonctionnalités principales

- Chargement de documents PDF financiers ou ESG
- Upload de documents via l'interface Streamlit
- Construction locale d'un index vectoriel FAISS
- Recherche sémantique des passages pertinents
- Génération de réponses avec un LLM local via Ollama
- Affichage des sources utilisées
- Affichage du nom du fichier, de la page et du score de similarité
- Possibilité de régler le nombre de passages récupérés avec le paramètre top-k
- Fonctionnement local, sans dépendance à une API payante

## Structure du projet

```text
financial-rag-assistant/
├── app.py
├── README.md
├── requirements.txt
├── src/
│   ├── ask.py
│   └── ingest.py
```

Les fichiers suivants sont volontairement exclus du dépôt GitHub :

```text
.env
.venv/
data/
vectorstore/
documents PDF
index FAISS générés
fichiers de cache
fichiers temporaires
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Smailoid/financial-rag-assistant.git
cd financial-rag-assistant
```

### 2. Créer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama

Installer Ollama localement, puis télécharger le modèle utilisé :

```bash
ollama pull llama3.2:3b
```

### 5. Ajouter des documents PDF

Créer localement le dossier suivant :

```text
data/reports/
```

Puis y placer les documents PDF à analyser.

Les documents PDF ne sont pas inclus dans le dépôt GitHub.

### 6. Construire l'index FAISS

```bash
python src/ingest.py
```

### 7. Lancer l'application Streamlit

```bash
streamlit run app.py
```

## Fonctionnement

Le pipeline suit les étapes suivantes :

1. Les documents PDF sont chargés depuis le dossier `data/reports/`.
2. Le texte est extrait à partir des PDF.
3. Les documents sont découpés en chunks.
4. Les chunks sont transformés en embeddings avec `all-MiniLM-L6-v2`.
5. Les embeddings sont stockés dans un index FAISS local.
6. Lorsqu'un utilisateur pose une question, les passages les plus proches sont récupérés.
7. Ces passages sont transmis comme contexte au modèle Llama 3.2 3B via Ollama.
8. Le modèle génère une réponse en français, accompagnée des sources utilisées.

## Limites du prototype

Ce projet est un prototype local destiné à l'expérimentation et à la démonstration.

Ses principales limites sont :

- Les PDF scannés ou constitués uniquement d'images ne sont pas correctement traités sans OCR.
- La qualité des réponses dépend de la qualité de l'extraction du texte.
- Le modèle Llama 3.2 3B est léger et local, mais moins performant qu'un modèle plus grand.
- L'évaluation des réponses est actuellement manuelle.
- Le système ne contient pas encore de pipeline d'évaluation automatique du RAG.
- Le projet ne comprend pas encore de déploiement cloud.
- Le projet ne comprend pas encore de CI/CD.
- Les documents uploadés et les index FAISS générés restent locaux et ne sont pas versionnés dans Git.

## Améliorations possibles

- Ajouter un module OCR pour les PDF scannés
- Mettre en place une évaluation automatique du retrieval et des réponses
- Comparer plusieurs modèles d'embeddings
- Ajouter du reranking
- Améliorer la gestion des erreurs dans l'interface
- Ajouter Docker
- Ajouter des tests automatisés
- Mettre en place une démo cloud

## Stack technique

Python, LangChain, Hugging Face Sentence Transformers, FAISS, Ollama, Llama 3.2 3B, Streamlit.
