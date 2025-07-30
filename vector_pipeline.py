# server/pipeline/vector_pipeline.py

from server.agents.extraction_agent import ExtractionAgent
from server.agents.chunking_agent import ChunkingAgent
from server.agents.embedding_agent import EmbeddingAgent
from server.agents.vectorstore_agent import VectorStoreAgent

class VectorPipeline:
    def __init__(self, kb_path: str, fi_path: str):
        self.kb_path = kb_path
        self.fi_path = fi_path

    def run(self):
        print("[VectorPipeline] Starting vector ingestion pipeline...")

        # Step 1: Extraction
        extractor = ExtractionAgent(self.kb_path, self.fi_path)
        kb_data = extractor.load_knowledge_bank()
        fi_data = extractor.load_field_issues()

        # Step 2: Chunking
        chunker = ChunkingAgent()
        chunks = chunker.run(kb_data, fi_data)

        # Step 3: Embedding
        embedder = EmbeddingAgent()
        embedded_chunks = embedder.embed_chunks_sync(chunks)

        # Step 4: Store in Qdrant
        vectorstore = VectorStoreAgent()
        vector_dim = len(embedded_chunks[0]["embedding"])
        vectorstore.create_collection(vector_dim)
        vectorstore.add_embeddings(embedded_chunks)

        return vectorstore.collection_name
