import sys
import numpy as np
from cortex import CortexClient, DistanceMetric
from cortex.filters import Filter, Field
import torch.nn.functional as F

from google import genai

import os
import json

from typing import Any

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

def get_titles():
    titles = []
    with open("articles_output/all_titles.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # remove whitespace and newlines
            if line:  # skip empty lines
                titles.append(line)

    return titles

def get_summaries():
    summaries = []
    with open("articles_output/all_summaries.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # remove whitespace and newlines
            if line:  # skip empty lines
                summaries.append(line)
                
    return summaries

def main():
    embeddings = create_embedding()
    titles = get_titles()
    summaries = get_summaries()
    ids = list(range(len(embeddings)))
    DOCUMENTS: list[dict[str, Any] | None] | None = [{"title": t, "summary": s} for t, s in zip(titles, summaries)]
    vectors = [vec for vec in embeddings]
    print("=" * 60)
    print("VectorDB Seeding")
    print("=" * 60)
    
    with CortexClient(SERVER) as client: # with makes sure this closes properly at the end of the with condition
        version, _ = client.health_check()
        print(f"\n✓ Connected to {version}")
        
        # Create collection
        print("\n1. Creating document collection...")
        client.create_collection(
            name=COLLECTION,
            dimension=DIMENSION,
            distance_metric=DistanceMetric.COSINE,
        )
        client.batch_upsert(COLLECTION, ids, vectors, DOCUMENTS)
        print("   ✓ Indexed")

if __name__ == "__main__":
    main()