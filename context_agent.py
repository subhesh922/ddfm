# server/agents/context_agent.py

import os
import re
import json
import asyncio
from typing import List, Dict
from server.agents.vectorstore_agent import VectorStoreAgent
from openai import AsyncAzureOpenAI

class ContextAgent:
    def __init__(self, collection_name: str, batch_size: int = 5):
        self.collection_name = collection_name
        self.vectorstore = VectorStoreAgent(collection_name=self.collection_name)
        self.batch_size = batch_size

        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

        self.system_msg = (
            "You are a DFMEA analyst with deep domain expertise in Enterprise Mobile Computing at Zebra Technologies.\n\n"

            "You are provided with raw structured text from two sources:\n"
            "- SOURCE: knowledge_bank — entries from the DFMEA Knowledge Bank\n"
            "- SOURCE: field_issues — real-world field failure reports containing the product type, fault_code, part_category (component) and fault_type\n\n"

            "Your job is to analyze and correlate both sources to produce DFMEA entries in the following strict JSON structure:\n\n"

            "Each DFMEA entry should include:\n"
            "- ID: Auto-incremented integer (start from 1)\n"
            "- Product: Product name from field_issues (e.g. 'TC57')\n"
            "- Subsystems: List of subsystems, each with:\n"
            "  - Subsystem: Name of subsystem (e.g. 'Display')\n"
            "  - Components: List of components, each with:\n"
            "    - Component: Component name (e.g. 'Touch Panel')\n"
            "    - Function: What the component does\n"
            "    - FailureModes: List of failure scenarios, each with:\n"
            "      - FailureMode: What failed\n"
            "      - Effects: List of effects, each with:\n"
            "        - Effect (string)\n"
            "        - Severity (integer 1–10)\n"
            "      - Causes: List of root causes, each with:\n"
            "        - Cause (string)\n"
            "        - Occurrence (1–10)\n"
            "        - Detection (1–10)\n"
            "        - Controls Prevention (list of 1–3 actionable items — must not be empty)\n"
            "        - Controls Detection (list of 1–3 techniques — must not be empty)\n"
            "        - Recommended Actions (list of 1–3 suggestions — must not be empty)\n"
            "        - RPN (calculated as Severity × Occurrence × Detection)\n"
            "        - linked_to_dfmea_kb (boolean — true if matched with knowledge bank)\n\n"

            "RULES:\n"
            "- Use only data from the chunked input text (SOURCE: ...).\n"
            "- Generate only for Display subsystem.\n"
            "- Return **pure valid JSON** (no explanations, no markdown, no triple backticks).\n"
            "- Start output with '[' and ensure the output can be parsed using `json.loads()`.\n"
            "- You can return 1–3 DFMEA entries per batch.\n"
            "- Create atleast 50 percent of the DFMEA entries as brand new from your intelligence of understanding the field_issues.\n"
            "- Ensure numeric fields are integers, and use empty lists `[]` for missing fields.\n"
            "- Do not leave Controls Prevention, Controls Detection, or Recommended Actions empty.\n"
            "- If unknown, infer based on failure context. Do not use generic statements. Stick to DFMEA best practices.\n"
            "- Do not provide redundant statements like implement DFMEA and stick to Hardware issues only."
        )

    def _batch(self, lst: List, n: int):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    async def _call_azure_openai(self, user_msg: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.3,
            messages=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return response.choices[0].message.content

    async def _parse_llm_response(self, raw_response: str) -> List[Dict]:
        cleaned = re.sub(r"^```(json)?", "", raw_response.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            print("[ContextAgent] JSON decode failed after cleanup.")
            return []

    async def _process_batch(self, batch_chunks: List[str], index: int) -> List[Dict]:
        print(f"[ContextAgent] Processing batch {index}...")
        user_msg = "Here are relevant data chunks:\n\n" + "\n\n".join(batch_chunks)

        for attempt in range(1, 3):
            try:
                raw_response = await self._call_azure_openai(user_msg)
                print(f"[ContextAgent] Raw LLM Response (batch {index}, attempt {attempt}):\n{raw_response[:300]}\n")

                parsed = await self._parse_llm_response(raw_response)
                if parsed:
                    return parsed
                else:
                    raise ValueError("Empty or invalid JSON after parsing.")

            except Exception as e:
                print(f"[ContextAgent] Batch {index} attempt {attempt} failed: {e}")
                await asyncio.sleep(1.5 * attempt)  # backoff

        print(f"[ContextAgent] Batch {index} permanently failed after 2 attempts.")
        return []

    async def run_async(self, query: str, top_k: int = 50) -> List[Dict]:
        print("[ContextAgent] Searching Qdrant for:", query)
        matches = self.vectorstore.search(query, top_k=top_k)
        chunks = [m["text"] for m in matches]

        print(f"[ContextAgent] Retrieved {len(chunks)} chunks from Qdrant.\n")
        tasks = [self._process_batch(batch, i) for i, batch in enumerate(self._batch(chunks, self.batch_size), 1)]

        results = await asyncio.gather(*tasks)
        flattened = [entry for batch in results for entry in batch]
        print(f"\n[ContextAgent] Parsed {len(flattened)} DFMEA entries.\n")
        return flattened

    def run(self, query: str, top_k: int = 50) -> List[Dict]:
        return asyncio.run(self.run_async(query, top_k=top_k))
