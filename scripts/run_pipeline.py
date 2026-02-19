import yaml
import os
import joblib

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.mining.clustering import EmployeeClustering
from src.mining.association import HRAssociationMining
from src.models.supervised import LogisticModel
from src.evaluation.metrics import classification_metrics
from src.evaluation.report import EvaluationReport
from sklearn.model_selection import train_test_split

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

MODEL_PATH = "outputs/models/attrition_rf.joblib"

df = DataLoader(cfg["paths"]["raw_data"]).load()

X, y, feature_names = DataCleaner().clean(df, cfg["target"])

clusterer = EmployeeClustering(
    n_clusters=cfg["clustering"]["n_clusters"],
    random_state=cfg["seed"]
)
cluster_labels, sil = clusterer.fit(X)
print("Silhouette score:", sil)

if os.path.exists(MODEL_PATH):
    print("Found trained model. Loading...")
    model = joblib.load(MODEL_PATH)
else:
    print("No trained model found. Training new model...")
    trainer = LogisticModel(cfg["seed"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg['seed'],
        stratify=y
    )
    trainer.train(X_train, y_train)
    model = trainer.model
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

metrics = classification_metrics(y, y_pred, y_prob)
print("Classification metrics:", metrics)

report = EvaluationReport(output_dir="outputs/tables")
report.add_result("RandomForest", metrics)
report.save_csv("evaluation_results.csv")

assoc = HRAssociationMining(
    min_support=cfg["association"]["min_support"],
    min_confidence=cfg["association"]["min_confidence"]
)

df_assoc = assoc.preprocess(df)
rules = assoc.mine_rules(df_assoc)

rules_path = "outputs/tables/association_rules.csv"
rules.to_csv(rules_path, index=False)

print(f"Association rules saved to {rules_path}")
