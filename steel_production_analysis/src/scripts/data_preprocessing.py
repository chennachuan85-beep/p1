"""
data_preprocessing.py
---------------------
Stage 2: Data preprocessing + splitting + normalization

Includes:
1. Duplicate removal
2. Missing value imputation (median)
3. Outlier detection and removal using IQR
4. Categorical variable encoding
5. Data consistency checks
"""

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler





#
# def split_and_normalize_data(
#     df,
#     target_column,
#     val_size=0.4,
#     test_size=0.0,
#     random_state=42
# ):
#     """
#     Split data into train / validation / test sets
#     Normalize features using StandardScaler
#     """
#
#     # =========================
#     # 1. Remove duplicate rows
#     # =========================
#     df = df.drop_duplicates()
#
#     # =========================
#     # 2. Split features / target
#     # =========================
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#
#     # =========================
#     # 3. Handle missing values
#     # =========================
#     # Separate numerical and categorical columns
#     numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
#     categorical_cols = X.select_dtypes(include=["object", "category"]).columns
#
#     # Numerical: median imputation
#     X[numerical_cols] = X[numerical_cols].fillna(
#         X[numerical_cols].median()
#     )
#
#     # Categorical: mode imputation (if any)
#     for col in categorical_cols:
#         X[col] = X[col].fillna(X[col].mode()[0])
#
#     # =========================
#     # 4. Encode categorical variables
#     # =========================
#     if len(categorical_cols) > 0:
#         X = pd.get_dummies(X, drop_first=True)
#
#     # =========================
#     # # 5. Outlier removal using IQR
#     # #    (applied before splitting)
#     # # =========================
#     # Q1 = X.quantile(0.25)
#     # Q3 = X.quantile(0.75)
#     # IQR = Q3 - Q1
#     #
#     # mask = ~(
#     #     (X < (Q1 - 1.5 * IQR)) |
#     #     (X > (Q3 + 1.5 * IQR))
#     # ).any(axis=1)
#     #
#     # X = X.loc[mask]
#     # y = y.loc[mask]
#
#     # =========================
#     # 6. Train / temp split
#     # =========================
#     if test_size > 0:
#     # 原有的三路划分逻辑
#         X_train, X_temp, y_train, y_temp = train_test_split(
#             X,
#             y,
#             test_size=val_size + test_size,
#             random_state=random_state
#         )
#
#     # =========================
#     # 7. Validation / test split
#     # =========================
#         val_ratio = val_size / (val_size + test_size)
#
#         X_val, X_test, y_val, y_test = train_test_split(
#             X_temp,
#             y_temp,
#             test_size=1 - val_ratio,
#             random_state=random_state
#         )
#     else:
#         # 当不需要测试集时，只进行一次划分，避免调用报错
#         X_train, X_val, y_train, y_val = train_test_split(
#             X,
#             y,
#             test_size=val_size,
#             random_state=random_state
#         )
#         # 创建空的 DataFrame/Series 作为占位符，保证返回结构一致
#         X_test = pd.DataFrame(columns=X.columns)
#         y_test = pd.Series(dtype=y.dtype)
#
#
#
#
#     # =========================
#     # NEW: Outlier removal (TRAIN ONLY)
#     # =========================
#     remove_outliers = True  # 🔧 可开关
#
#     if remove_outliers:
#         Q1 = X_train.quantile(0.25)
#         Q3 = X_train.quantile(0.75)
#         IQR = Q3 - Q1
#
#         mask = ~(
#                 (X_train < (Q1 - 1.5 * IQR)) |
#                 (X_train > (Q3 + 1.5 * IQR))
#         ).any(axis=1)
#
#         X_train = X_train.loc[mask]
#         y_train = y_train.loc[mask]
#
#
#
#
#
#     # =========================
#     # 8. Normalization
#     # =========================
#     scaler = StandardScaler()
#
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     # 检查 X_test 是否为空，避免 StandardScaler 报错
#     if not X_test.empty:
#         X_test_scaled = scaler.transform(X_test)
#     else:
#         X_test_scaled = X_test.values  # 保持为 numpy 格式以匹配后续转换
#     #X_test_scaled = scaler.transform(X_test)
#
#     X_train_scaled = pd.DataFrame(
#         X_train_scaled, columns=X_train.columns, index=X_train.index
#     )
#     X_val_scaled = pd.DataFrame(
#         X_val_scaled, columns=X_val.columns, index=X_val.index
#     )
#     X_test_scaled = pd.DataFrame(
#         X_test_scaled, columns=X_test.columns, index=X_test.index
#     )
#
#     # =========================
#     # 9. Data consistency checks
#     # =========================
#     assert X_train_scaled.shape[0] == y_train.shape[0], "Train data mismatch"
#     assert X_val_scaled.shape[0] == y_val.shape[0], "Validation data mismatch"
#     assert X_test_scaled.shape[0] == y_test.shape[0], "Test data mismatch"
#
#     return (
#         X_train_scaled,
#         X_val_scaled,
#         X_test_scaled,
#         y_train,
#         y_val,
#         y_test,
#         scaler
#     )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_and_normalize_data(
    df,
    target_column,
    val_size=0.2,
    test_size=0.0,
    random_state=42
):
    """
    Split data into train/validation/test sets
    Normalize features using StandardScaler
    """

    # =========================
    # 1. Feature / target split
    # =========================
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # =========================
    # 2. Train / temp split
    # =========================
    if test_size > 0:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=val_size + test_size,
            random_state=random_state
        )

        val_ratio = val_size / (val_size + test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_ratio,
            random_state=random_state
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=random_state
        )
        X_test = pd.DataFrame(columns=X.columns)
        y_test = pd.Series(dtype=y.dtype)

    # =========================
    # 3. Normalization
    # =========================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if not X_test.empty:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    # 转回 DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        X_val_scaled, columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    # =========================
    # 4. Consistency checks
    # =========================
    assert X_train_scaled.shape[0] == y_train.shape[0]
    assert X_val_scaled.shape[0] == y_val.shape[0]
    assert X_test_scaled.shape[0] == y_test.shape[0]

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler
    )
