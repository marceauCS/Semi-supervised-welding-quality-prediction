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
import xgboost as xgb

def fast_forward_selection_manual_cv(df,
                                    numerical_cols,
                                    onehot_cols,
                                    target_name,
                                    regressor,
                                    params,
                                    max_features=15,
                                    n_splits=5):
    """
    Fast forward feature selection with manual K-Fold cross-validation.
    Designed for speed by using fewer estimators and avoiding complex pipelines.

    Note
    ----
    Data leakage is intentionally allowed here to accelerate the feature selection process.
    This function is intended only to rank features quickly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target.
    numerical_cols : list
        List of numerical feature column names.
    onehot_cols : list
        List of one-hot encoded feature column names.
    target_name : str
        Name of the target column.
    regressor : estimator class
        Regression model class (e.g., XGBRegressor, RandomForestRegressor).
    params : dict
        Parameters to initialize the regressor.
    max_features : int, default=15
        Maximum number of features to select.
    n_splits : int, default=5
        Number of folds for manual cross-validation.

    Returns
    -------
    selected_features : list
        List of selected feature names in the order they were added.
    """

    # Combine all candidate features
    all_features = numerical_cols + onehot_cols

    # Filter out rows with missing target
    y = df[target_name].values
    mask = ~np.isnan(y)
    y_filtered = y[mask]

    X_full = df[all_features].values[mask]

    # Standardize features
    X_full_scaled = StandardScaler().fit_transform(X_full)

    selected, remaining = [], list(range(len(all_features)))
    results = []
    mean_target = np.mean(y_filtered)

    kf = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_full_scaled))

    while remaining and len(selected) < max_features:
        best_rmse, best_idx = np.inf, None

        for idx in remaining:
            feats = selected + [idx]
            model = regressor(**params)
            # --- boucle manuelle KFold ---
            rmses = []
            for train_idx, test_idx in kf:
                X_train, X_test = X_full_scaled[train_idx][:, feats], X_full_scaled[test_idx][:, feats]
                y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mean_rmse = np.mean(rmses)

            if mean_rmse < best_rmse:
                best_rmse, best_idx = mean_rmse, idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        results.append({
            'step': len(selected),
            'added_feature': all_features[best_idx],
            'RMSE': best_rmse,
            'RMSE_pct': best_rmse / mean_target * 100
        })
        print(f"Step {len(selected):2d} | Added: {all_features[best_idx]:30s} | RMSE={best_rmse:.2f} ({best_rmse/mean_target*100:.1f}%)")

    return [all_features[i] for i in selected]