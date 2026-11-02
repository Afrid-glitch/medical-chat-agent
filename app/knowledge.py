
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


# MiniLM produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384


# Load the local embedding model once
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Initialize Pinecone client
@lru_cache()
def clients():
    if not settings.pinecone_api_key:
        raise ValueError(
            "Pinecone API key must be set in your .env file"
        )

    return Pinecone(api_key=settings.pinecone_api_key)


# Create or connect to Pinecone index
def ensure_index():
    pc = clients()

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    index = pc.Index(settings.pinecone_index)

    # Check whether the index dimension matches MiniLM
    stats = index.describe_index_stats()

    if stats.dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Pinecone index dimension is {stats.dimension}, "
            f"but the embedding model requires "
            f"{EMBEDDING_DIMENSION}. Please create a new "
            f"384-dimensional Pinecone index."
        )

    return index


# Generate embeddings locally
def embed(texts):
    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return vectors.tolist()


# Search relevant information from the knowledge base
def search_kb(query, top_k=5):
    index = ensure_index()

    # Generate query embedding using the same local model
    vector = embed([query])[0]

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "page": match.metadata.get("page"),
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 3),
            "source": "Gale Encyclopedia (2002)",
        }
        for match in result.matches
        if match.score >= 0.30
    ]
