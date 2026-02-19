# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# class AttritionModel:
#     def __init__(self, params, seed=42):
#         self.model = RandomForestClassifier(
#             **params,
#             random_state=seed
#         )

#     def train(self, X, y, test_size):
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y,
#             test_size=test_size,
#             random_state=42,
#             stratify=y
#         )
#         self.model.fit(X_train, y_train)
#         return X_train, X_test, y_train, y_test

#     def predict(self, X):
#         return self.model.predict(X)

#     def predict_proba(self, X):
#         return self.model.predict_proba(X)[:, 1]


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class BaseModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LogisticModel(BaseModel):
    def __init__(self, seed=42):
        model = LogisticRegression(max_iter=1000, random_state=seed)
        super().__init__(model)


class RandomForestModel(BaseModel):
    def __init__(self, params, seed=42):
        model = RandomForestClassifier(
            **params,
            random_state=seed
        )
        super().__init__(model)


class XGBoostModel(BaseModel):
    def __init__(self, params, seed=42):
        model = XGBClassifier(
            **params,
            random_state=seed,
            eval_metric="logloss"
        )
        super().__init__(model)

