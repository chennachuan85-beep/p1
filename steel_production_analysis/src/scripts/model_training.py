"""
model_training.py
-----------------
Stage 4: Model training with hyperparameter tuning
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from scripts.data_loading import load_data
from scripts.data_preprocessing import split_and_normalize_data

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# =========================
# Load data
# =========================
train_df, test_df = load_data()
target_column = train_df.columns[0]  # ✅ 正确的 output

#full_df = pd.concat([train_df, test_df], axis=0)

#X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_normalize_data(
#    full_df, target_column
#)
X_train, X_val, _, y_train, y_val, _, _ = split_and_normalize_data(
    train_df,
    target_column,
    val_size=0.2,
    test_size=0.0
)

# =========================
# Tuned models
# =========================
models = {
    "RandomForest":
       # RandomForestRegressor(
       #  max_depth=12,
       # min_samples_split=5,
       # min_samples_leaf=2,
       # random_state=42
    #),
     RandomForestRegressor(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,        #"sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),


    "SVR":
        #SVR(
        #kernel="rbf",
        #C=10,
        #gamma="scale",
        #epsilon=0.05
    #),
    SVR(
        kernel="rbf",
        C=1.0,
        gamma=0.1,
        epsilon=0.01,
        degree=3
    ),



    "MLP":
        #MLPRegressor(
        #hidden_layer_sizes=(128, 64),
        #activation="relu",
        #alpha=1e-4,
        #learning_rate_init=0.001,
        #max_iter=2000,
        #random_state=42
    #),
    MLPRegressor(
        hidden_layer_sizes=(420, 336, 252, 168),
        activation="relu",
        solver="adam",
        alpha=1e-6,
        learning_rate_init=0.01,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
        learning_rate="constant",
        batch_size="auto"
    ),





    "GaussianProcess": GaussianProcessRegressor(
        kernel=C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0,
            length_scale_bounds=(1e-5, 1e2)
        ),
        alpha=1e-6,
        optimizer="fmin_l_bfgs_b",
        normalize_y=True,
        random_state=42
        )
}


# =========================
# Validation evaluation
# =========================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    results.append({
        "Model": name,
        "Val_RMSE": rmse,
        "Val_R2": r2
    })

    print(f"{name} | Validation RMSE={rmse:.4f}, R2={r2:.4f}")

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/validation_performance_tuned.csv", index=False)
