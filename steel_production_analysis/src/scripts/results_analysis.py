"""
results_analysis.py
-------------------
Stage 5: Final evaluation with proper use of validation set

This script:
1. Trains models on training set
2. Selects best model based on VALIDATION performance
3. Evaluates best model once on TEST set
4. Computes RMSE, MAE, R2
5. Measures training and inference time
6. Generates all required plots (PDF 4.2)
"""

# =========================
# Step 1: Imports
# =========================
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from scripts.data_loading import load_data
from scripts.data_preprocessing import split_and_normalize_data


# =========================
# Step 2: Visualization functions (PDF 4.2)
# =========================
# Setup base directories
base_dir = os.path.dirname(os.path.dirname(__file__))
result_dir = os.path.join(os.path.dirname(base_dir), "result")
figures_dir = os.path.join(result_dir, "figures", "model")
table_dir = os.path.join(result_dir, "table")


def plot_model_comparison(results_df):
    """Create bar plots with error bars"""
    plt.figure(figsize=(8, 5))
    plt.bar(
        results_df["Model"],
        results_df["Val_RMSE"],
        yerr=results_df["Val_RMSE"].std(),
        capsize=6
    )
    plt.ylabel("Validation RMSE")
    plt.title("Model Comparison (Validation RMSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison_validation_rmse.png"))
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    """Scatter plot of predictions vs actual values"""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{model_name}_pred_vs_actual.png"))
    plt.close()


def plot_residuals(y_true, y_pred, model_name):
    """Plot residual analysis"""
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({model_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{model_name}_residuals.png"))
    plt.close()


def plot_learning_curve(model, X, y, model_name):
    """Plot learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_rmse, label="Training RMSE")
    plt.plot(train_sizes, val_rmse, label="Validation RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{model_name}_learning_curve.png"))
    plt.close()


# =========================
# Step 3: Load and prepare data
# =========================
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)
os.makedirs(os.path.join(table_dir, "model_predictions"), exist_ok=True)

train_df, test_df = load_data()
target_column = train_df.columns[0]

X_train, X_val, _, y_train, y_val, _, scaler = \
    split_and_normalize_data(
        train_df,
        target_column,
        test_size=0.0
    )



# =========================
# Build INDEPENDENT test set
# =========================

# 1. 特征 / 标签分离
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# 2. 分类变量编码（必须与训练一致）
X_test = pd.get_dummies(X_test, drop_first=True)

# 3. 特征对齐（极其关键）
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 4. 使用训练集 scaler 进行归一化
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns,
    index=X_test.index
)

# 5. 安全检查（强烈建议）
assert X_test.shape[0] > 0, "Independent test set is empty!"





# =========================
# Step 4: Define models
# =========================
models = {
    "RandomForest": RandomForestRegressor(
       # n_estimators=200,
       # random_state=42
    ),
    "SVR": SVR(),
    "MLP": MLPRegressor(
      # max_iter=500,
       # random_state=42
    ),
    "GaussianProcess": GaussianProcessRegressor(
       # kernel=RBF()
    )
}


# =========================
# Step 5: Validation-based model selection
# =========================
validation_results = []

trained_models = {}

for name, model in models.items():
    print(f"\nTraining & validating model: {name}")

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    validation_results.append({
        "Model": name,
        "Val_RMSE": val_rmse,
        "Training_Time_sec": train_time
    })

    trained_models[name] = model

    # Learning curve (uses training data only)
    plot_learning_curve(model, X_train, y_train, name)

    print(f"{name} | Validation RMSE: {val_rmse:.4f}")


validation_df = pd.DataFrame(validation_results)
validation_df.to_csv(
    os.path.join(table_dir, "validation_performance.csv"),
    index=False
)

plot_model_comparison(validation_df)


# =========================
# Step 6: Select best model
# =========================
best_model_name = validation_df.sort_values("Val_RMSE").iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\n✅ Best model selected based on validation RMSE: {best_model_name}")








# =========================
# OPTIONAL: Evaluate ALL models on TEST set
# =========================
test_results_all_models = []

for name, model in trained_models.items():
    print(f"Evaluating {name} on TEST set")

    start_pred = time.time()
    y_test_pred_all = model.predict(X_test)
    inference_time = time.time() - start_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_all))
    mae = mean_absolute_error(y_test, y_test_pred_all)
    r2 = r2_score(y_test, y_test_pred_all)

    test_results_all_models.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Inference_Time_sec": inference_time
    })

# Save all-model test results
test_results_df = pd.DataFrame(test_results_all_models)
test_results_df.to_csv(
    os.path.join(table_dir, "test_performance_all_models.csv"),
    index=False
)

print(f"All-model test performance saved to {os.path.join(table_dir, 'test_performance_all_models.csv')}")

# =========================
# Step 7: Final evaluation on TEST set
# =========================
start_pred = time.time()
y_test_pred = best_model.predict(X_test)
inference_time = time.time() - start_pred

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

final_results = pd.DataFrame([{
    "Best_Model": best_model_name,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2,
    "Inference_Time_sec": inference_time
}])

final_results.to_csv(
    os.path.join(table_dir, "final_test_performance.csv"),
    index=False
)

# Save predictions
pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_test_pred
}).to_csv(
    os.path.join(table_dir, "model_predictions", f"{best_model_name}_predictions.csv"),
    index=False
)

# Required plots on TEST set
plot_predictions_vs_actual(y_test, y_test_pred, best_model_name)
plot_residuals(y_test, y_test_pred, best_model_name)


print("\n🎉 Stage 5 completed with proper validation usage.")
print(f"Validation results: {os.path.join(table_dir, 'validation_performance.csv')}")
print(f"Final test results: {os.path.join(table_dir, 'final_test_performance.csv')}")
