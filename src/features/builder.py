"""
Feature engineering cho HR Analytics.
Tạo các đặc trưng mới từ dữ liệu đã tiền xử lý.
"""
import numpy as np
import pandas as pd


class FeatureBuilder:
    def build(self, X, feature_names=None):

        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=feature_names or [f"f_{i}" for i in range(X.shape[1])])
        else:
            df = X.copy()

        new_cols = []

        if "YearsAtCompany" in df.columns and "TotalWorkingYears" in df.columns:
            denom = df["TotalWorkingYears"].replace(0, np.nan)
            df["TenureRatio"] = (df["YearsAtCompany"] / denom).fillna(0)
            new_cols.append("TenureRatio")

        if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
            denom = df["YearsAtCompany"].replace(0, np.nan)
            df["PromotionDelay"] = (df["YearsSinceLastPromotion"] / denom).fillna(0)
            new_cols.append("PromotionDelay")

        if "YearsAtCompany" in df.columns and "NumCompaniesWorked" in df.columns:
            df["JobStability"] = df["YearsAtCompany"] / (df["NumCompaniesWorked"] + 1)
            new_cols.append("JobStability")

        feature_names_new = df.columns.tolist()
        return df.values, feature_names_new
