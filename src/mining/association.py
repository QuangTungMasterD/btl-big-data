import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class HRAssociationMining:
    def __init__(self, min_support=0.05, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def preprocess(self, df):
        df_bin = df.copy()

        df_bin["IncomeLevel"] = pd.qcut(
            df_bin["MonthlyIncome"],
            q=3,
            labels=["LowIncome", "MidIncome", "HighIncome"]
        )

        df_bin["TenureLevel"] = pd.qcut(
            df_bin["YearsAtCompany"],
            q=3,
            labels=["ShortTenure", "MidTenure", "LongTenure"]
        )

        cols = ["OverTime", "Attrition", "IncomeLevel", "TenureLevel"]
        df_bin = df_bin[cols]

        df_bin = pd.get_dummies(df_bin)
        return df_bin

    def mine_rules(self, df):
        freq = apriori(
            df,
            min_support=self.min_support,
            use_colnames=True
        )

        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=self.min_confidence
        )

        rules = rules.sort_values("lift", ascending=False)
        return rules
