import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


# ==============================
# EDA
# ==============================

def plot_attrition_distribution(series, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=series)
    plt.title("Attrition Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_income_by_attrition(df, save_path=None):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
    plt.title("Monthly Income by Attrition")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_numeric_histograms(df, save_path=None):
    num_cols = df.select_dtypes(include="number").columns

    df[num_cols].hist(
        figsize=(15, 10),
        bins=20,
        layout=(len(num_cols)//4 + 1, 4)
    )

    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_correlation_matrix(df, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_numeric_boxplots(df, save_path=None):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(12, 6))
    df[num_cols].boxplot()
    plt.xticks(rotation=45)
    plt.title("Boxplot of Numeric Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


# ==============================
# MODELING
# ==============================

def plot_model_comparison(results_df, metric="F1", save_path=None):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y=metric)

    plt.ylabel(metric)
    plt.title(f"Model Comparison ({metric})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_confusion(y_true, y_pred, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


# ==============================
# CLUSTERING
# ==============================

def plot_cluster_pca(X_pca, labels, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=20)
    plt.title("PCA visualization of clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_elbow(K, inertias, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(K, inertias, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_missing_values(missing_series, save_path=None):
    """
    missing_series: pandas Series
        index = tên cột
        value = số lượng missing
    """
    plt.figure(figsize=(8, 4))

    missing_series.sort_values(ascending=False).plot(kind="bar")

    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_scaling_comparison(before, after, feature, save_path=None):
    """
    So sánh phân phối 1 feature trước và sau khi scaling
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(before[feature], kde=True)
    plt.title("Before Scaling")

    plt.subplot(1, 2, 2)
    sns.histplot(after[feature], kde=True)
    plt.title("After Scaling")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_learning_curve(results_df, save_path=None):
    plt.figure(figsize=(6,4))

    plt.plot(
        results_df["ratio"] * 100,
        results_df["sup_F1"],
        marker="o",
        label="Supervised"
    )

    plt.plot(
        results_df["ratio"] * 100,
        results_df["semi_F1"],
        marker="o",
        label="Semi-supervised"
    )

    plt.xlabel("Tỷ lệ dữ liệu có nhãn (%)")
    plt.ylabel("F1-score")
    plt.title("Learning curve: Supervised vs Semi-supervised")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()

