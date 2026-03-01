import sys
import numpy as np
from cortex import CortexClient, DistanceMetric
from cortex.filters import Filter, Field
import torch.nn.functional as F

from google import genai

import os
import json

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import requests
import urllib.parse

from eventregistry import *
import time, datetime

# Configuration, if user provided an argument or not for a different URL
SERVER = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
# Use unique name to avoid  file conflicts
COLLECTION = "News_Articles"
DIMENSION = 384  # We are using E5-small-v2

def create_embedding():
    titles = []
    with open("articles_output/all_titles.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # remove whitespace and newlines
            if line:  # skip empty lines
                titles.append(line)

    model = SentenceTransformer("intfloat/e5-small-v2")
    embeddings = model.encode(titles, normalize_embeddings=True) # normalize_embeddings scales each vector to have a magnitude of 1
    return embeddings

def main():
    embeddings = create_embedding()
    print("=" * 60)
    print("Cortex Semantic Search Example")
    print("=" * 60)
    
    with CortexClient(SERVER) as client: # with makes sure this closes properly at the end of the with condition
        version, _ = client.health_check()
        print(f"\n✓ Connected to {version}")
        
        # Create collection
        print(f"\n1. Creating document collection...")
        client.create_collection(
            name=COLLECTION,
            dimension=DIMENSION,
            distance_metric=DistanceMetric.COSINE,
        )
        
        # Index documents
        print(f"\n2. Indexing {len(DOCUMENTS)} documents...")
        ids = list(range(len(DOCUMENTS)))
        vectors = [simulate_embedding(doc["title"]) for doc in DOCUMENTS]
        
        client.batch_upsert(COLLECTION, ids, vectors, DOCUMENTS)
        print(f"   ✓ Indexed {len(DOCUMENTS)} documents")
        
        # Basic search
        print("\n" + "-" * 40)
        query = "machine learning models"
        print(f"Query: '{query}'")
        query_vec = simulate_embedding(query)
        
        results = client.search(COLLECTION, query_vec, top_k=5)
        print("\nTop 5 results:")
        for i, r in enumerate(results):
            vec, payload = client.get(COLLECTION, r.id)
            print(f"  {i+1}. {payload['title']} ({payload['category']}, {payload['year']}) - Score: {r.score:.4f}")
        
        # Category-specific search
        print("\n" + "-" * 40)
        print("Query: 'database' filtered by category='Database'")
        
        query_vec = simulate_embedding("database")
        f = Filter().must(Field("category").eq("Database"))
        
        results = client.search_filtered(COLLECTION, query_vec, f, top_k=3)
        print(f"\nDatabase documents ({len(results)} results):")
        for i, r in enumerate(results):
            vec, payload = client.get(COLLECTION, r.id)
            print(f"  {i+1}. {payload['title']} ({payload['year']})")
        
        # Scroll all documents
        print("\n" + "-" * 40)
        print("Scrolling all documents by category:")
        
        categories = {}
        next_cursor = None
        while True:
            records, next_cursor = client.scroll(COLLECTION, limit=10, cursor=next_cursor)
            for record in records:
                cat = record.payload.get("category", "Unknown") if record.payload else "Unknown"
                categories[cat] = categories.get(cat, 0) + 1
            if next_cursor is None:
                break
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} documents")
        
        # Cleanup
        print("\n3. Cleanup...")
        client.delete_collection(COLLECTION)
        print(f"   ✓ Collection deleted")
    
    print("\n" + "=" * 60)
    print("✓ Semantic Search Example Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()