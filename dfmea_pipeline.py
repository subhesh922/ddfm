# server/pipeline/dfmea_pipeline.py

from server.agents.context_agent import ContextAgent
from server.agents.writer_agent import WriterAgent
from crewai import Agent, Task, Crew

class DFMEAPipeline:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def run(self, query="Generate DFMEA entries for recent field failures", top_k=100):
        print("[DFMEAPipeline] Starting DFMEA generation pipeline...")

        # Step 1: Contextual reasoning using RAG
        context = ContextAgent(collection_name=self.collection_name, batch_size=5)
        structured_json = context.run(query=query, top_k=top_k)

        # Step 2: Write output to Excel + JSON
        writer = WriterAgent()
        output_path = writer.run(structured_json)

        return output_path

    # Optional: Crew trace
    def crew ():
        crew = Crew(
            agents=[
                Agent(name="Reasoning LLM", description="Generate DFMEA structured output."),
                Agent(name="Writer", description="Writes structured DFMEA JSON to Excel.")
            ],
            tasks=[
                Task(description="Query embeddings and return JSON format."),
                Task(description="Write DFMEA results.")
            ],
            verbose=True
        )

        
