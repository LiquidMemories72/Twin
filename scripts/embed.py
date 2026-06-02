import json
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

# ----------------------------
# CONFIG
# ----------------------------

QDRANT_URL = "https://f8f50026-5a88-4fa6-b0f6-08b72729a789.australia-southeast1-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Yzg4YjliYmQtZTI1OS00YmM0LWE1ZWUtOTg2N2RjMjRiNTYxIn0.fEqk86YEMzOx71lNNNl8mU5FQQZuCQGyOEF3Pbmupn4"

COLLECTION_NAME = "Twin"

# ----------------------------
# LOAD MODEL
# ----------------------------

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ----------------------------
# CONNECT
# ----------------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ----------------------------
# CREATE COLLECTION
# ----------------------------

collections = client.get_collections()

existing = [
    c.name
    for c in collections.collections
]

if COLLECTION_NAME not in existing:

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Collection created")

# ----------------------------
# LOAD CHUNKS
# ----------------------------

with open(
    "data/all_chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ----------------------------
# EMBED + UPLOAD
# ----------------------------

batch = []
batch_size = 64

for idx, chunk in enumerate(chunks):

    vector = model.encode(
        chunk["text"],
        normalize_embeddings=True
    ).tolist()

    payload = chunk.copy()

    batch.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload=payload
        )
    )

    if len(batch) >= batch_size:

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

        print(
            f"Uploaded {idx+1}/{len(chunks)}"
        )

        batch = []

# final batch

if batch:

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

print("Upload complete")