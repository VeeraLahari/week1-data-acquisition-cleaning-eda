import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Data acquisition
iris = load_iris(as_frame=True)
df = iris.frame.copy()

df.columns = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "target"
]

df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=["target"])

# 2. Initial inspection
print(df.head())
print("Shape:", df.shape)
print(df.info())
print("Missing values:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# 3. Cleaning
df = df.drop_duplicates().reset_index(drop=True)

numeric_cols = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["species"] = df["species"].astype("category")

# 4. EDA
print(df.describe())
print(df.groupby("species")[numeric_cols].mean())
print("Correlation:")
print(df[numeric_cols].corr())

# 5. Visualizations
df.isnull().sum().plot(kind="bar", title="Missing Values by Column")
plt.tight_layout()
plt.show()

df["species"].value_counts().sort_index().plot(kind="bar", title="Iris Class Distribution")
plt.tight_layout()
plt.show()

for species in df["species"].cat.categories:
    subset = df[df["species"] == species]
    plt.scatter(subset["petal_length_cm"], subset["petal_width_cm"], label=species)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")
plt.legend()
plt.tight_layout()
plt.show()

df[numeric_cols].boxplot()
plt.title("Numeric Feature Distributions")
plt.tight_layout()
plt.show()
