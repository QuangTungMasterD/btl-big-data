import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class EmployeeClustering:
    def __init__(self, n_clusters=4, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X):
        labels = self.model.fit_predict(X)
        score = silhouette_score(X, labels)
        return labels, score

    def profile_clusters(self, df_original, labels):
        df = df_original.copy()
        df["Cluster"] = labels
        profile = df.groupby("Cluster").mean(numeric_only=True)
        return profile
