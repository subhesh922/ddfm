import os
import uuid
import math
from typing import List, Dict
from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

class VectorStoreAgent:
    def __init__(self, collection_name: str = None):
        self.qdrant_url = os.getenv("QDRANT_ENDPOINT")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.base_collection = os.getenv("QDRANT_COLLECTION", "dfmea_collection")
        self.session_id = str(uuid.uuid4())[:8]
        self.collection_name = collection_name or f"{self.base_collection}_{self.session_id}"
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            prefer_grpc=False,  # Make HTTP-based async fallback smoother
            https=True,
            timeout=30,
            verify=False
        )
        # Disable SSL verification for all requests
        self.ssl_verify = False

    def create_collection(self, vector_dim: int):
        print(f"[VectorStoreAgent] Creating collection '{self.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print(f"[VectorStoreAgent] Created session collection: {self.collection_name}")

    def add_embeddings(self, embedded_chunks: List[Dict], batch_limit: int = 500):
        print(f"[VectorStoreAgent] Uploading {len(embedded_chunks)} vectors in batches...")

        points = [
            PointStruct(
                id=idx,
                vector=chunk["embedding"],
                payload={**chunk.get("metadata", {}), "text": chunk["text"]}
            )
            for idx, chunk in enumerate(embedded_chunks)
        ]

        # Split into batches to avoid 32MB payload limits
        num_batches = math.ceil(len(points) / batch_limit)
        for i in range(num_batches):
            batch = points[i * batch_limit : (i + 1) * batch_limit]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
            except Exception as e:
                print(f"[VectorStoreAgent] Batch {i+1} failed: {type(e).__name__}: {e}")
        print("[VectorStoreAgent] Upload complete.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        print(f"[VectorStoreAgent] Searching for: '{query}' in '{self.collection_name}'")

        embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        response = embedding_client.embeddings.create(input=query, model=deployment)
        query_vector = response.data[0].embedding

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        output = []
        for hit in results:
            output.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload
            })

        print(f"[VectorStoreAgent] Found {len(output)} matches.")
        return output

    def delete_collection(self):
        print(f"[VectorStoreAgent] Dropping collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        print("[VectorStoreAgent] Collection deleted.")
