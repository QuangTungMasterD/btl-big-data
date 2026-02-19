import papermill as pm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    "01_eda.ipynb",
    "02_preprocess_feature.ipynb",
    "03_clustering_mining.ipynb",
    "04_modeling.ipynb",
    "04b_semi_supervised.ipynb",
    "05_evaluation_report.ipynb",
]

EXECUTED_DIR = PROJECT_ROOT / "notebooks" / "_executed_"
EXECUTED_DIR.mkdir(parents=True, exist_ok=True)

for nb in NOTEBOOKS:
    print(f"Running {nb} ...")
    pm.execute_notebook(
        input_path=PROJECT_ROOT / "notebooks" / nb,
        output_path=EXECUTED_DIR / nb,
        parameters={
            "project_root": str(PROJECT_ROOT)
        },
        kernel_name=None
    )

print("All notebooks executed successfully.")
