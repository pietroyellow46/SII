# 🏀 NBA Knowledge Graph & RAG Agent

Sistema intelligente per la ricerca e raccomandazione di giocatori NBA basato su:

- 🤖 LLM (LLaMA-3 + Groq)
- 🔎 Vector Search (Weaviate + SentenceTransformers)
- 🕸️ Graph Analysis (NetworkX + Louvain)

Un sistema RAG che combina statistiche NBA, ricerca semantica e analisi dei grafi.

---

## ✨ Features

### 🔍 Natural Language Search

Cerca giocatori usando linguaggio naturale:


Trovami ali piccole con tanti rimbalzi e oltre il 38% da tre


L'LLM estrae filtri statistici e contesto semantico per interrogare il database.

---

### 👥 Player Recommendation

Trova giocatori simili alle leggende NBA tramite:

- Similarità statistica
- Similarità di stile di gioco
- Community detection sul grafo NBA

---

### 🕸️ NBA Knowledge Graph

Costruisce una rete di giocatori:

- nodi → giocatori
- archi → similarità

Export disponibile:


data/exports/graph_nba.graphml


---

## 🛠️ Tech Stack

- LLaMA-3 + Groq
- Weaviate
- SentenceTransformers
- NetworkX
- Louvain
- Streamlit

---

## 📂 Structure

```text
NBA-Knowledge-Graph-RAG/

│
├── src/
│   ├── app.py
│   ├── main.py
│   ├── query_LLM.py
│   ├── sii.py
│   └── preprocessing.py
│
├── data/
│   ├── raw/
│   ├── exports/
│   │   └── graph_nba.graphml
│   │
│   └── dataset_nba_completo.json
│
├── notebooks/
│   └── scraping + EDA
│
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```



---

## ⚙️ Setup

Requirements:

- Python >= 3.9
- Docker
- Groq API Key

Install:

```bash
git clone https://github.com/username/repo.git
cd repo
pip install -r requirements.txt
```

Create .env:
```bash
GROQ_API_KEY=your_key
```

Avvio:

```bash
docker-compose up -d
python .\src\main.py
streamlit run src/app.py
```

---