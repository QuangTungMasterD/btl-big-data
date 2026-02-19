import pandas as pd
import os

class EvaluationReport:
    def __init__(self, output_dir="outputs/tables"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def add_result(self, model_name, metrics_dict):
        row = {"Model": model_name}
        row.update(metrics_dict)
        self.results.append(row)

    def to_dataframe(self):
        return pd.DataFrame(self.results)

    def save_csv(self, filename="evaluation_results.csv"):
        df = self.to_dataframe()
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        return path
