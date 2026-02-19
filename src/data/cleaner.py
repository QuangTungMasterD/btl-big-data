import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.builder import FeatureBuilder


class DataCleaner:
    def clean(self, df, target):
        df = df.copy()

        df = df.drop_duplicates()

        drop_cols = [
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Over18"
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        y = df[target].map({"Yes": 1, "No": 0})
        X = df.drop(columns=[target])

        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns

        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        for col in num_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower, upper)

        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        X = X.astype(float)

        X_arr, feature_names = FeatureBuilder().build(X.values, X.columns.tolist())
        return X_arr, y, feature_names
