# server/agents/writer_agent.py

import os
import json
import pandas as pd
from typing import List, Dict

class WriterAgent:
    def __init__(self, output_path: str = "output/dfmea_output.xlsx", json_dump_path: str = "output/raw_dfmea_output.json"):
        self.output_path = output_path
        self.json_dump_path = json_dump_path

    def _flatten_dfmea(self, dfmea_json: List[Dict]) -> pd.DataFrame:
        """
        Flatten the hierarchical DFMEA JSON into tabular structure.
        """
        rows = []
        unique_id = 1

        for entry in dfmea_json:
            product = entry.get("Product", "Unknown Product")
            subsystems = entry.get("Subsystems", [])

            for sub in subsystems:
                subsystem = sub.get("Subsystem", "")
                components = sub.get("Components", [])

                for comp in components:
                    component = comp.get("Component", "")
                    function = comp.get("Function", "")
                    failure_modes = comp.get("FailureModes", [])

                    for fm in failure_modes:
                        failure_mode = fm.get("FailureMode", "")
                        effects = fm.get("Effects", [])
                        causes = fm.get("Causes", [])

                        for effect in effects:
                            effect_text = effect.get("Effect", "")
                            severity = effect.get("Severity", "")

                            for cause in causes:
                                row = {
                                    "ID": unique_id,
                                    "Product": product,
                                    "Subsystem": subsystem,
                                    "Component": component,
                                    "Function": function,
                                    "Failure Mode": failure_mode,
                                    "Effect": effect_text,
                                    "Severity": severity,
                                    "Cause": cause.get("Cause", ""),
                                    "Occurrence": cause.get("Occurrence", ""),
                                    "Detection": cause.get("Detection", ""),
                                    "Controls Prevention": ", ".join(cause.get("Controls Prevention", [])),
                                    "Controls Detection": ", ".join(cause.get("Controls Detection", [])),
                                    "Recommended Actions": ", ".join(cause.get("Recommended Actions", [])),
                                    "RPN": cause.get("RPN", ""),
                                    "Exists in DFMEA KB": cause.get("linked_to_dfmea_kb", False)
                                }
                                rows.append(row)
                                unique_id += 1

        return pd.DataFrame(rows)

    def run(self, dfmea_json: List[Dict]) -> str:
        print(f"[WriterAgent] Writing DFMEA output to: {self.output_path}")

        # Save raw JSON for debugging
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.json_dump_path, "w") as f:
            json.dump(dfmea_json, f, indent=2)
        print(f"[WriterAgent] Raw DFMEA JSON dumped to: {self.json_dump_path}")

        # Flatten and write Excel
        df = self._flatten_dfmea(dfmea_json)
        df.to_excel(self.output_path, index=False)
        print(f"[WriterAgent] Excel file saved with {len(df)} rows.\n")

        return self.output_path
