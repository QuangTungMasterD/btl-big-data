from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
import numpy as np

def mask_labels(y, labeled_ratio, seed=42):
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(y)) < labeled_ratio

    y_masked = y.copy()
    y_masked[~mask] = -1
    return y_masked, mask


class SemiSupervisedAttrition:
    def __init__(self, seed=42):
        base = RandomForestClassifier(random_state=seed)
        self.model = SelfTrainingClassifier(base)

    def train(self, X, y_partial):
        self.model.fit(X, y_partial)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
