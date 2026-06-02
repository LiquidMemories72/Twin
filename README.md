# Yann LeCun Digital Twin

A Retrieval-Augmented Generation (RAG) based digital twin of AI pioneer Yann LeCun. The application allows users to converse with a virtual representation of Yann LeCun, which provides answers grounded in his research papers, interviews, talks, and transcripts. 

## Features

- **Conversational Interface:** Built with [Streamlit](https://streamlit.io/), facilitating a chat-like experience.
- **RAG Architecture:** Uses Qdrant vector database for retrieving relevant papers and transcripts.
- **Advanced Retrieval:** Implements basic retrieval via `BAAI/bge-small-en-v1.5` embeddings followed by reranking with `BAAI/bge-reranker-base`.
- **Generative AI:** Powered by Google's `gemini-2.5-flash` model, configured to adopt Yann LeCun's technical style, perspectives, and principles.
- **Source Transparency:** Highlights retrieval metrics, scores, and full context sources used to answer each query.

## Project Structure

- `Project/`: Contains the main application code (e.g., `app2.py`).
- `scripts/`: Contains scripts for downloading, embedding, chunking, and managing the AI context data (papers, transcripts).
- `data/`: Local storage for papers, JSON chunks, and transcripts (ignored by default in version control).
- `qdrant_db/`: Vector database storage (ignored by default in version control).

## Setup & Installation

### 1. Prerequisites 

Make sure you have Python installed and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install the dependencies:

```bash
pip install streamlit qdrant-client sentence-transformers python-dotenv google-generativeai
```

### 2. Environment Variables

Create a `.env` file either in the root directory or inside the `Project/` folder with the following variables:

```ini
GEMINI_API_KEY=your_google_gemini_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Data Processing (Optional)

If starting from scratch without an existing `qdrant_db`, you will need to run the data collection, chunking, and embedding scripts located in the `scripts/` directory to populate the vector database with Yann LeCun's content.

### 4. Running the Application

To start the Streamlit application, run:

```bash
streamlit run Project/app2.py
```

The application will open in your browser, where you can begin asking questions!
