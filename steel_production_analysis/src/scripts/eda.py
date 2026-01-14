"""
eda.py
------
Stage 3: Exploratory Data Analysis
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.data_loading import load_data


train_df, _ = load_data()
target_column = train_df.columns[0]

X = train_df.drop(columns=[target_column])
y = train_df[target_column]

base_dir = os.path.dirname(os.path.dirname(__file__))
figures_dir = os.path.join(os.path.dirname(base_dir), "result", "figures", "eda")
os.makedirs(figures_dir, exist_ok=True)

# Target distribution
plt.figure()
sns.histplot(y, kde=True)
plt.title("Target Distribution")
plt.savefig(os.path.join(figures_dir, "target_distribution.png"))
plt.close()

# Histograms
for col in X.columns:
    plt.figure()
    sns.histplot(X[col], kde=True)
    plt.title(col)
    plt.savefig(os.path.join(figures_dir, f"{col}_hist.png"))
    plt.close()

# Boxplots
for col in X.columns:
    plt.figure()
    sns.boxplot(x=X[col])
    plt.title(col)
    plt.savefig(os.path.join(figures_dir, f"{col}_box.png"))
    plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
plt.close()

# Pair plot (subset)
pair_df = X.iloc[:, :5].copy()
pair_df["target"] = y
sns.pairplot(pair_df)
plt.savefig(os.path.join(figures_dir, "pairplot.png"))
plt.close()

print("EDA completed.")
