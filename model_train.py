import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from preprocess import *
import xgboost as xgb

def get_onehot_cols(features):
    """
    Identify columns created through one-hot encoding.
    One-hot encoded columns are detected by the presence of an underscore ('_') in their names.

    Parameters
    ----------
    features : list or pd.Index
        List of feature names.

    Returns
    -------
    list
        Columns likely representing one-hot encoded features.
    """
    return [col for col in features if "_" in col]


def get_numerical_cols(features):
    """
    Identify numerical (non-one-hot encoded) feature columns.

    Parameters
    ----------
    features : list or pd.Index
        List of feature names.

    Returns
    -------
    list
        Columns that are not one-hot encoded.
    """
    onehot_cols = get_onehot_cols(features)
    return [col for col in features if col not in onehot_cols]



def train_regressor_target(
    df,
    target_name,
    numerical_cols,
    onehot_cols,
    regressor,
    params,
    target_cols=None,
    n_splits=10,
    plot=True
):
    """
    Train a regression model for a specific target using custom imputation,
    feature scaling, and K-Fold cross-validation.

    Missing value strategy:
    ------------------------
    - Sulphur & Phosphorus concentration → IterativeImputer (mean strategy)
    - Other concentration columns → filled with 0
    - Other numerical features → IterativeImputer (median strategy)
    - One-hot encoded columns → filled with 0

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target.
    target_name : str
        Name of the target column to predict.
    numerical_cols : list
        List of numerical feature columns.
    onehot_cols : list
        List of one-hot encoded feature columns.
    regressor : estimator class
        Regression model class (e.g., XGBRegressor, RandomForestRegressor, LinearRegressor...).
    params : dict
        Parameters to initialize the regressor.
    target_cols : list, optional
        List of target column names (used to exclude from feature sets during imputation).
    n_splits : int, default=10
        Number of folds for cross-validation.
    plot : bool, default=True
        If True, generate performance visualizations.

    Returns
    -------
    rmse : float
        Root Mean Squared Error across all folds.
    corr : float
        Pearson correlation between true and predicted values.
    relative_rmse : float
        RMSE expressed as a percentage of the mean true value.
    """

    # --- Prepare target ---
    y_df = df[[target_name]]
    mask = ~y_df[target_name].isna()
    y_filtered = y_df.loc[mask, target_name].reset_index(drop=True)

    X_filtered = df.loc[mask, numerical_cols + onehot_cols].reset_index(drop=True)
    X_train_test = X_filtered.copy()

    # --- Cross-validation setup ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_list, y_true_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_train_test)):
        X_train, X_test = X_train_test.iloc[train_idx], X_train_test.iloc[test_idx]
        y_train, y_test = y_filtered.iloc[train_idx], y_filtered.iloc[test_idx]

        # --- Impute missing values ---
        X_train, X_test = impute_missing_values(X_train, X_test, target_cols or [])

        # --- Standardize numerical features ---
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[numerical_cols]),
            columns=numerical_cols,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[numerical_cols]),
            columns=numerical_cols,
            index=X_test.index
        )

        # Combine scaled numerical features with one-hot features
        X_train_scaled = pd.concat([X_train_scaled, X_train[onehot_cols]], axis=1)
        X_test_scaled = pd.concat([X_test_scaled, X_test[onehot_cols]], axis=1)

        # --- Train model ---
        model = regressor(**params)
        model.fit(X_train_scaled, y_train.values.ravel())
        y_pred = model.predict(X_test_scaled)

        y_pred_list.append(y_pred)
        y_true_list.append(y_test.values)

    # --- Aggregate results ---
    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    corr = np.corrcoef(y_true_all, y_pred_all)[0, 1]
    relative_rmse = round(float((rmse / np.mean(y_true_all)) * 100), 2)

    # --- Visualization ---
    if plot:
        plt.figure(figsize=(7, 5))
        sns.kdeplot(y_true_all, label="True", fill=True, alpha=0.4)
        sns.kdeplot(y_pred_all, label="Predicted", fill=True, alpha=0.4)
        plt.title(f"Distribution Comparison – {target_name}")
        plt.legend()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_all, y_pred_all, alpha=0.5)
        plt.plot(
            [y_true_all.min(), y_true_all.max()],
            [y_true_all.min(), y_true_all.max()],
            'r--'
        )
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{target_name}: True vs Predicted")
        plt.show()

    return (rmse, corr,relative_rmse)



def benchmark_ineq_strategies(df_origin, target_cols, best_features, xgb_params, ineq_strategies):
    """
    Benchmark different inequality-handling strategies across all target variables,
    returning a summary of model performance for each combination.

    The function iterates over all targets and evaluates multiple inequality 
    preprocessing strategies using cross-validation.

    Parameters
    ----------
    df_origin : pd.DataFrame
        Original dataset before preprocessing.
    target_cols : list
        List of target column names to model.
    best_features : dict
        Dictionary mapping each target to its selected feature set.
    xgb_params : dict
        Dictionary of hyperparameters for the XGBoost regressor.
    ineq_strategies : list
        List of inequality handling strategies to test 
        (e.g., ["half", "minus10", "zero"]).

    Returns
    -------
    pd.DataFrame
        Summary DataFrame containing model performance for each target and strategy.
        Columns include:
        - target
        - ineq_strategy
        - corr (correlation coefficient)
        - rmse (root mean squared error)
        - rmse_rel (RMSE as percentage of mean true value)
    """

    results_summary = []

    for target in target_cols:
        # Select features corresponding to the target
        numerical_cols_reduced = get_numerical_cols(best_features[target])
        onehot_cols_reduced = get_onehot_cols(best_features[target])

        for strat in ineq_strategies:
            # --- Full preprocessing using the selected inequality strategy ---
            df_filled = preprocess_data(df_origin, ineq_strategy=strat)

            # --- Model training and evaluation ---
            rmse, corr, rmse_rel = train_regressor_target(
                df=df_filled,
                target_name=target,
                numerical_cols=numerical_cols_reduced,
                onehot_cols=onehot_cols_reduced,
                plot=False,
                regressor=XGBRegressor,
                params=xgb_params,
                n_splits=10
            )

            # --- Store results ---
            results_summary.append({
                "target": target,
                "ineq_strategy": strat,
                "corr": corr,
                "rmse": rmse,
                "rmse_rel": rmse_rel
            })

    # Combine results into a DataFrame sorted by target and descending correlation
    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values(["target", "corr"], ascending=[True, False]).reset_index(drop=True)

    return df_results



def fill_missing_targets(df_filled, target_configs, impute_missing_values):
    """
    Semi-supervised filling of missing targets using regression models.

    Parameters
    ----------
    df_filled : pd.DataFrame
        Dataset with missing target values.
    target_configs : dict
        Dictionary defining features and model parameters per target.
    impute_missing_values : function
        Function used to impute missing feature values.

    Returns
    -------
    df_final : pd.DataFrame
        Dataset with all missing targets filled.
    predicted_indices : dict
        Dictionary with target names as keys and indices of predicted rows as values.
    summary_df : pd.DataFrame
        Summary table of filled counts and distribution stats.
    """

    df_final = df_filled.copy()
    filled_counts = []
    predicted_indices = {}

    print("\n=== Final Training and Prediction Phase ===")

    for target_name, cfg in target_configs.items():
        print(f"\n───────────────────────────────────────────────")
        print(f"Target: {target_name}")
        print("───────────────────────────────────────────────")

        numerical_cols = cfg["numerical_cols"]
        onehot_cols = cfg["onehot_cols"]
        params = cfg["params"]
        regressor = cfg["regressor"]

        # --- Select feature columns ---
        feature_cols = [c for c in numerical_cols + onehot_cols if c in df_final.columns]

        # --- Split train / predict ---
        mask_train = ~df_final[target_name].isna()
        mask_pred = df_final[target_name].isna()
        n_train, n_pred = mask_train.sum(), mask_pred.sum()

        print(f"Training samples: {n_train}")
        print(f"Missing samples to predict: {n_pred}")

        if n_train == 0:
            print(f"Skipping {target_name}: no labeled data available.")
            continue
        if n_pred == 0:
            print(f"Skipping {target_name}: no missing values to fill.")
            continue

        # --- Prepare features ---
        X_train = df_final.loc[mask_train, feature_cols].copy()
        y_train = df_final.loc[mask_train, target_name].copy()
        X_pred = df_final.loc[mask_pred, feature_cols].copy()

        # --- Impute missing features ---
        X_train_imp, X_pred_imp = impute_missing_values(X_train, X_pred, target_cols=[target_name])

        # --- One-hot safety ---
        if onehot_cols:
            X_train_imp[onehot_cols] = X_train_imp[onehot_cols].fillna(0).astype(float)
            X_pred_imp[onehot_cols] = X_pred_imp[onehot_cols].fillna(0).astype(float)

        # --- Standardize ---
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=X_train_imp.columns,
            index=X_train_imp.index
        )
        X_pred_scaled = pd.DataFrame(
            scaler.transform(X_pred_imp),
            columns=X_pred_imp.columns,
            index=X_pred_imp.index
        )

        # --- Train model ---
        model = regressor(**params)
        model.fit(X_train_scaled, y_train)

        # --- Predict ---
        y_pred_missing = model.predict(X_pred_scaled)
        df_final.loc[mask_pred, target_name] = y_pred_missing
        predicted_indices[target_name] = df_final.loc[mask_pred].index.tolist()

        print(f"Filled {n_pred} missing values for {target_name}.")

        # --- Store info ---
        filled_counts.append({
            "target": target_name,
            "n_filled": n_pred,
            "n_train": n_train,
            "train_mean": y_train.mean(),
            "pred_mean": np.mean(y_pred_missing),
            "train_std": y_train.std(),
            "pred_std": np.std(y_pred_missing)
        })

        # --- Visualization ---
        plt.figure(figsize=(10, 4))
        sns.kdeplot(y_train, label="Observed", fill=True, alpha=0.4)
        sns.kdeplot(y_pred_missing, label="Predicted", fill=True, alpha=0.4)
        plt.title(f"Distribution Comparison — {target_name}")
        plt.xlabel(target_name)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=[y_train, y_pred_missing])
        plt.xticks([0, 1], ["Observed", "Predicted"])
        plt.title(f"Boxplot Comparison — {target_name}")
        plt.tight_layout()
        plt.show()

    # --- Summary report ---
    summary_df = pd.DataFrame(filled_counts)
    print("\n\n=== Summary of Filled Targets ===")
    display(summary_df)

    # --- Summary plot ---
    if not summary_df.empty:
        plt.figure(figsize=(9, 5))
        sns.scatterplot(
            data=summary_df,
            x="train_mean",
            y="pred_mean",
            hue="target",
            s=100
        )
        plt.plot(
            [summary_df["train_mean"].min(), summary_df["train_mean"].max()],
            [summary_df["train_mean"].min(), summary_df["train_mean"].max()],
            "--", color="gray"
        )
        plt.title("Observed vs Predicted Mean Values per Target")
        plt.xlabel("Observed Mean")
        plt.ylabel("Predicted Mean")
        plt.tight_layout()
        plt.show()
    else:
        print("No targets were filled — all complete or skipped.")

    return df_final, predicted_indices, summary_df