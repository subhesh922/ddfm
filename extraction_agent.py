# server/agents/extraction_agent.py

from pathlib import Path
from server.utils.excel_parser import parse_excel_or_csv

class ExtractionAgent:
    def __init__(self, kb_path=None, fi_path=None):
        base_dir = Path(__file__).resolve().parent.parent

        # Use default sample files if none provided
        self.kb_path = Path(kb_path) if kb_path else base_dir / "sample_files" / "dfmea_knowledge_bank_3.csv"
        self.fi_path = Path(fi_path) if fi_path else base_dir / "sample_files" / "field_reported_issues_3.xlsx"

        # Validate file paths
        if not self.kb_path.is_file():
            raise FileNotFoundError(f"Knowledge Bank file not found: {self.kb_path}")
        if not self.fi_path.is_file():
            raise FileNotFoundError(f"Field Issues file not found: {self.fi_path}")

    def load_knowledge_bank(self):
        print(f"[ExtractionAgent] Loading DFMEA Knowledge Bank: {self.kb_path}")
        return parse_excel_or_csv(str(self.kb_path))

    def load_field_issues(self):
        print(f"[ExtractionAgent] Loading Field Reported Issues: {self.fi_path}")
        return parse_excel_or_csv(str(self.fi_path))
