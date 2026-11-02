
import argparse
import hashlib
import time
from pathlib import Path

import pymupdf

from app.knowledge import ensure_index, embed


# Configuration
BATCH_SIZE = 10
BATCH_DELAY = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 60


def chunks(text, size=1100, overlap=180):
    """Split page text into overlapping chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    step = size - overlap

    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if len(text[i:i + size]) > 80
    ]


def generate_embedding_with_retry(texts):
    """Generate embeddings with retry handling for rate limits."""

    for attempt in range(MAX_RETRIES):
        try:
            vectors = embed(texts)

            if len(vectors) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, "
                    f"received {len(vectors)}."
                )

            return vectors

        except Exception as exc:
            error_message = str(exc).upper()

            is_rate_limit = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "TOO MANY REQUESTS" in error_message
            )

            if not is_rate_limit:
                raise

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    "Gemini embedding quota is still unavailable "
                    "after multiple retries. Wait for the quota to "
                    "reset, then run this script again."
                ) from exc

            wait_seconds = INITIAL_RETRY_DELAY * (2 ** attempt)

            print(
                f"Gemini rate limit reached. "
                f"Retry {attempt + 1}/{MAX_RETRIES - 1} "
                f"in {wait_seconds} seconds.",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Embedding generation failed.")


def upload_batch(index, batch):
    """Embed a batch and upload it to Pinecone."""

    if not batch:
        return

    texts = [item[1] for item in batch]

    vectors = generate_embedding_with_retry(texts)

    records = [
        {
            "id": item[0],
            "values": vector,
            "metadata": item[2],
        }
        for item, vector in zip(batch, vectors)
    ]

    # Deterministic IDs allow existing records to be updated.
    index.upsert(vectors=records)

    print(
        f"Uploaded {len(records)} chunks to Pinecone.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a medical PDF into Pinecone."
    )

    parser.add_argument(
        "--pdf",
        default="data/medical_book.pdf",
        help="Path to the medical PDF file.",
    )

    args = parser.parse_args()
    pdf = Path(args.pdf)

    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")

    print(f"Opening PDF: {pdf}", flush=True)

    index = ensure_index()

    total_chunks = 0
    uploaded_chunks = 0
    batch = []

    try:
        with pymupdf.open(pdf) as doc:
            total_pages = len(doc)

            if total_pages == 0:
                raise SystemExit("The PDF contains no pages.")

            print(
                f"Total pages: {total_pages}",
                flush=True,
            )

            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text()

                for part, content in enumerate(
                    chunks(page_text), start=1
                ):
                    # Stable ID for each PDF chunk.
                    chunk_id = hashlib.sha256(
                        (
                            f"{pdf.name}:{page_number}:"
                            f"{part}:{content}"
                        ).encode("utf-8")
                    ).hexdigest()

                    metadata = {
                        "page": page_number,
                        "text": content,
                        "file": pdf.name,
                    }

                    batch.append(
                        (chunk_id, content, metadata)
                    )

                    total_chunks += 1

                    if len(batch) >= BATCH_SIZE:
                        upload_batch(index, batch)

                        uploaded_chunks += len(batch)
                        batch = []

                        print(
                            f"Progress: {uploaded_chunks} chunks "
                            f"uploaded; page {page_number}/"
                            f"{total_pages}.",
                            flush=True,
                        )

                        time.sleep(BATCH_DELAY)

                if page_number % 50 == 0:
                    print(
                        f"Processed pages: "
                        f"{page_number}/{total_pages}",
                        flush=True,
                    )

            # Upload the final partial batch.
            if batch:
                upload_batch(index, batch)
                uploaded_chunks += len(batch)
                batch = []

        print("\nIngestion completed successfully.", flush=True)
        print(f"Total chunks processed: {total_chunks}", flush=True)
        print(f"Chunks uploaded this run: {uploaded_chunks}", flush=True)

    except KeyboardInterrupt:
        print(
            "\nIngestion interrupted. Successfully uploaded "
            "Pinecone records remain stored.",
            flush=True,
        )
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\nIngestion failed: {exc}",
            flush=True,
        )
        print(
            "Check your API quota, embedding configuration, "
            "and Pinecone index settings before retrying.",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
