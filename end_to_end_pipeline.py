from server.pipeline.vector_pipeline import VectorPipeline
from server.pipeline.dfmea_pipeline import DFMEAPipeline

class DFMEAEndToEndPipeline:
    def __init__(self, kb_path: str, fi_path: str, query: str = "Generate DFMEA entries for recent field failures", top_k: int = 100):
        self.kb_path = kb_path
        self.fi_path = fi_path
        self.query = query
        self.top_k = top_k

    def run(self):
        print("[End-to-End] Starting full DFMEA pipeline...")

        # Step 1: Chunk, embed and store in Qdrant
        vector_pipeline = VectorPipeline(self.kb_path, self.fi_path)
        collection_name = vector_pipeline.run()

        # Step 2: Use context agent + writer
        dfmea_pipeline = DFMEAPipeline(collection_name)
        output_path = dfmea_pipeline.run(query=self.query, top_k=self.top_k)

        return output_path
