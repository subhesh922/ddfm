import os
import asyncio
import time
from typing import List, Dict
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import AzureOpenAI, RateLimitError, APIConnectionError, InternalServerError
from dotenv import load_dotenv
from tiktoken import get_encoding

load_dotenv()

class EmbeddingAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        )
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.batch_size = 50  # Tune for performance vs. rate limits
        self.cooldown = 2      # Cooldown in seconds between batches

    def _log_token_usage(self, embedded: List[Dict]):
        total_tokens = sum(chunk.get("tokens", 0) for chunk in embedded)
        kb_tokens = sum(chunk.get("tokens", 0) for chunk in embedded if chunk.get("metadata", {}).get("source") == "knowledge_bank")
        fi_tokens = sum(chunk.get("tokens", 0) for chunk in embedded if chunk.get("metadata", {}).get("source") == "field_issues")
        print("\n🔍 [EmbeddingAgent] Token Usage Summary:")
        print(f"  • Total Chunks Embedded: {len(embedded)}")
        print(f"  • Total Tokens Used: {total_tokens:,}")
        print(f"  • Knowledge Bank → {kb_tokens:,} tokens")
        print(f"  • Field Issues   → {fi_tokens:,} tokens")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, InternalServerError)),
        reraise=True
    )
    def _embed_batch_with_retry(self, batch_texts):
        return self.client.embeddings.create(input=batch_texts, model=self.deployment)

    def embed_chunks_sync(self, chunks: List[Dict]) -> List[Dict]:
        print(f"[EmbeddingAgent] 🚀 Embedding {len(chunks)} chunks with sync batching...")

        embedded_chunks = []
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="[EmbeddingAgent] Embedding"):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            try:
                response = self._embed_batch_with_retry(batch_texts)
                for idx, item in enumerate(response.data):
                    embedded_chunks.append({
                        "text": batch[idx]["text"],
                        "embedding": item.embedding,
                        "metadata": batch[idx].get("metadata", {}),
                        "tokens": self._count_tokens(batch[idx]["text"])
                    })
            except Exception as e:
                print(f"[EmbeddingAgent] Batch failed after retries: {type(e).__name__}: {e}")
            time.sleep(self.cooldown)  # Cooldown between batches

        self._log_token_usage(embedded_chunks)
        return embedded_chunks

    def _count_tokens(self, text: str) -> int:
        tokenizer = get_encoding("cl100k_base")
        return len(tokenizer.encode(text))
