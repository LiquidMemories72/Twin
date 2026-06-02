# Yann LeCun Digital Twin

A Retrieval-Augmented Generation (RAG) system that simulates how Yann LeCun would answer questions using his published papers, interviews, talks, and transcripts.

The project combines semantic retrieval, cross-encoder reranking, conversational memory, and Gemini 2.5 Flash to create a grounded conversational digital twin.

---

## Features

* Conversational chat interface built with Streamlit
* Retrieval-Augmented Generation (RAG)
* Qdrant Cloud vector database
* BGE-small dense embeddings
* BGE Cross-Encoder reranking
* Multi-turn conversation memory
* Source attribution for every answer
* Retrieval debugging panel
* Timeline-aware context using document metadata
* Grounded responses based on Yann LeCun's published work and public statements

---

## Architecture

```text
User Question
      ↓
BGE-small-en-v1.5 Embedding
      ↓
Qdrant Vector Search (Top 40)
      ↓
BGE-reranker-base Cross Encoder
      ↓
Top 10 Retrieved Chunks
      ↓
Prompt Construction
      ↓
Gemini 2.5 Flash
      ↓
Answer + Sources
```

---

## Dataset

The knowledge base contains:

* Yann LeCun research papers
* Interview transcripts
* Conference talks
* Public discussions and podcasts

Each chunk is stored with metadata:

```json
{
  "text": "...",
  "title": "...",
  "source_type": "paper",
  "year": 2024
}
```

This metadata is injected into the prompt so the model can reason about chronology, source type, and evolving viewpoints.

---

## Retrieval Pipeline

### Stage 1: Dense Retrieval

**Embedding Model**

```text
BAAI/bge-small-en-v1.5
```

The user query is converted into a normalized dense vector and searched against the Qdrant collection.

Top 40 candidates are retrieved.

### Stage 2: Cross-Encoder Reranking

**Reranker**

```text
BAAI/bge-reranker-base
```

Each retrieved chunk is paired with the user query and scored by a cross-encoder.

The top 10 highest-scoring chunks are selected for final context construction.

This significantly improves retrieval precision compared to pure vector search.

---

## Persona Design

The system is designed to reflect recurring themes in Yann LeCun's work:

* Self-supervised learning
* World models
* Predictive representations
* Reasoning and planning
* Scientific skepticism
* Limitations of purely autoregressive language models

The assistant is instructed to:

* Ground answers in retrieved sources
* Distinguish between evidence and inference
* Avoid hallucinating facts or quotes
* Remain concise and technical
* Explain concepts using Yann's documented reasoning style

---

## Memory

The system maintains short-term conversational memory using Streamlit session state.

The last six conversation turns are included in every prompt to support multi-turn discussions and follow-up questions.

---

## Retrieval Debug Panel

The application includes a debugging interface that displays:

* Number of papers retrieved
* Number of transcripts retrieved
* Reranker scores
* Vector similarity scores
* Source metadata
* Full retrieved chunk text
* Full context sent to Gemini

This helps evaluate retrieval quality and answer grounding.

---

## Technologies Used

| Component              | Technology             |
| ---------------------- | ---------------------- |
| Frontend               | Streamlit              |
| LLM                    | Gemini 2.5 Flash       |
| Embeddings             | BAAI/bge-small-en-v1.5 |
| Reranker               | BAAI/bge-reranker-base |
| Vector Database        | Qdrant Cloud           |
| Environment Management | python-dotenv          |

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd yann-lecun-digital-twin
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=<your-key>

QDRANT_URL=https://f8f50026-5a88-4fa6-b0f6-08b72729a789.australia-southeast1-0.gcp.cloud.qdrant.io

QDRANT_API_KEY=<provided-qdrant-api-key>
```

Run the application:

```bash
streamlit run app.py
```

---

## Qdrant Configuration

**Collection Name**

```text
Twin
```

**Evaluation Qdrant URL**

```text
<INSERT_QDRANT_URL_HERE>
```

**Evaluation Qdrant API Key**

```text
<INSERT_EVALUATION_API_KEY_HERE>
```

> Note: Use the evaluation credentials provided with the submission package. Do not commit credentials to GitHub.

---



## Future Improvements

* Metadata-aware retrieval filters
* Timeline-specific search
* Reflection-based answer verification
* LangGraph orchestration
* Hybrid lexical + vector retrieval
* Long-term memory summarization

---

## Author

Built as a learning project exploring:

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Vector Databases
* Conversational AI
* Digital Twin Systems
