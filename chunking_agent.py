# server/agents/chunking_agent.py

import uuid
import tiktoken
import warnings
from tqdm import tqdm
from typing import List, Dict

class ChunkingAgent:
    def __init__(self, max_tokens=1500, overlap=100, model_name="text-embedding-ada-002"):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.overlap = overlap

        warnings.filterwarnings("ignore", category=UserWarning)

    def run(self, kb_data: List[Dict], fi_data: List[Dict]) -> List[Dict]:
        print(f"[ChunkingAgent] Chunking knowledge bank...")
        kb_chunks = self._create_chunks(kb_data, source="knowledge_bank")

        print(f"[ChunkingAgent] Chunking field-reported issues...")
        fi_chunks = self._create_chunks(fi_data, source="field_issues")

        all_chunks = kb_chunks + fi_chunks
        print(f"[ChunkingAgent] Merged into {len(all_chunks)} smart chunks before token slicing.")

        sliced_chunks = self._token_slice_chunks(all_chunks)
        return sliced_chunks

    def _create_chunks(self, data: List[Dict], source: str) -> List[Dict]:
        chunks = []
        for row in data:
            text = self._format_row_as_text(row)
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {
                        "uuid": str(uuid.uuid4()),
                        "source": source
                    }
                })
        return chunks

    def _format_row_as_text(self, row: Dict) -> str:
        return " | ".join(
            f"{k.strip()}: {str(v).strip()}"
            for k, v in row.items()
            if v and str(v).strip()
        )

    def _token_slice_chunks(self, chunks: List[Dict]) -> List[Dict]:
        sliced_chunks = []
        total_tokens = 0  # Initialize a counter to accumulate total tokens

        for chunk in tqdm(chunks, desc="[ChunkingAgent] Token slicing"):
            tokens = self.encoder.encode(chunk["text"])
            total_tokens += len(tokens)  # Accumulate the number of tokens for this chunk

            if len(tokens) <= self.max_tokens:
                sliced_chunks.append(chunk)
                continue

            start = 0
            while start < len(tokens):
                end = min(start + self.max_tokens, len(tokens))
                token_slice = tokens[start:end]
                text_slice = self.encoder.decode(token_slice)

                sliced_chunks.append({
                    "text": text_slice,
                    "metadata": chunk["metadata"]
                })

                if end == len(tokens):
                    break
                start += self.max_tokens - self.overlap

        print(f"[ChunkingAgent] Tokens sliced into {len(sliced_chunks)} total chunks.")
        print(f"[ChunkingAgent] Total tokens across all chunks: {total_tokens}")  # Print the total token count

        return sliced_chunks

