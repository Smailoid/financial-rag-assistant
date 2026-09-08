# Financial RAG Assistant

Projet de chatbot RAG permettant d'interroger des documents financiers PDF à l'aide d'un LLM, d'embeddings et d'une base vectorielle FAISS.

## Objectif

Construire un assistant capable de :
- charger des rapports financiers au format PDF ;
- découper les documents en chunks ;
- créer des embeddings ;
- stocker les embeddings dans une base FAISS locale ;
- répondre à des questions en citant les sources utilisées.

## Stack technique

- Python
- LangChain
- OpenAI API
- FAISS
- PyPDF
- dotenv

## Structure du projet

```text
financial-rag-assistant/
│
├── data/
│   └── reports/
│
├── src/
│   ├── ingest.py
│   └── ask.py
│
├── vectorstore/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md