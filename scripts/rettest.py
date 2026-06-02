from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "https://f8f50026-5a88-4fa6-b0f6-08b72729a789.australia-southeast1-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Yzg4YjliYmQtZTI1OS00YmM0LWE1ZWUtOTg2N2RjMjRiNTYxIn0.fEqk86YEMzOx71lNNNl8mU5FQQZuCQGyOEF3Pbmupn4"

COLLECTION_NAME = "Twin"

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

query = "What is JEPA?"

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5
)

for point in results.points:

    print("=" * 80)

    print(
        point.payload["title"]
    )

    print()

    print(
        point.payload["text"][:500]
    )