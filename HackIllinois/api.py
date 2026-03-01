from typing import Optional
from fastapi import FastAPI, Query

from cortex import AsyncCortexClient

import asyncio

import sys
from cortex import CortexClient, DistanceMetric

from typing import Any

from sentence_transformers import SentenceTransformer

from eventregistry import *

DIMENSION = 384  # We are using E5-small-v2

SERVER = "localhost:50051"
COLLECTION = "News_Articles"
app = FastAPI()

async def get_embedded_query(query_text: str):
    async with AsyncCortexClient(SERVER) as client:
        # Example: embedding and then finding closest vectors
        model = SentenceTransformer("intfloat/e5-small-v2")
        embedded_query = model.encode(
            query_text, normalize_embeddings=True, convert_to_numpy=True
        )

        results = await client.search(collection_name=COLLECTION, query=embedded_query, top_k=1,with_payload=True, with_vectors=False)
        print(results)
        return results

# Accept a query parameter ?sentence=...
@app.get("/closest_article")
async def closest_article(sentence: str = Query()): # pyright: ignore[reportArgumentType]
    results = await get_embedded_query(sentence)
    return {"results": results}