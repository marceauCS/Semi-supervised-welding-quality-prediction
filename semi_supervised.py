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
from preprocess import *

def fill_target_with_xgb_confidence(
    df, target_name, numerical_cols, onehot_cols, regressor, params,
    random_state=42, confidence_quantile=0.25, return_model=False
):
    """
    Fill NaN values in `target_name` using a regression model (default: XGBRegressor),
    selecting only the most "confident" predictions based on the cumulative variance
    of predictions from the trees.

    Parameters
    ----------
    df : DataFrame
        Dataset containing the target and features.
    target_name : str
        Name of the target column to impute.
    numerical_cols : list
        List of numerical columns.
    onehot_cols : list, optional
        List of one-hot encoded columns.
    regressor : estimator
        Regressor class to use (default: xgb.XGBRegressor).
    params : dict
        Hyperparameters to pass to the regressor.
    random_state : int, optional
        Random seed for reproducibility.
    confidence_quantile : float, optional
        Quantile (between 0 and 1) defining the confidence threshold for keeping pseudo-labels.
    return_model : bool
        If True, also returns the trained model.

    Returns
    -------
    df_modified : DataFrame
        DataFrame with NaNs filled for the target column.
    model : trained model (optional)
        Returns the trained model if `return_model=True`.
    """

    if onehot_cols is None:
        onehot_cols = []

    if params is None:
        params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.6,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=random_state,
            n_jobs=-1
        )

    # --- Split train/predict ---
    mask_train = ~df[target_name].isna()
    mask_pred = df[target_name].isna()
    n_to_fill = mask_pred.sum()
    if n_to_fill == 0:
        print(f"{target_name}: no NaN values to fill.")
        return (df, None) if return_model else df

    # --- Select available columns ---
    cols_num = [c for c in numerical_cols if c in df.columns]
    cols_onehot = [c for c in onehot_cols if c in df.columns]
    feature_cols = cols_num + cols_onehot

    X_train_raw = df.loc[mask_train, feature_cols].copy()
    X_pred_raw = df.loc[mask_pred, feature_cols].copy()
    y_train = df.loc[mask_train, target_name].copy()

    # --- 1️⃣ Full imputation of features ---
    X_train_imputed, X_pred_imputed = impute_missing_values(
        X_train_raw, X_pred_raw, target_cols=[target_name]
    )

    # --- 2️⃣ One-hot safety ---
    if cols_onehot:
        X_train_imputed[cols_onehot] = X_train_imputed[cols_onehot].fillna(0).astype(float)
        X_pred_imputed[cols_onehot] = X_pred_imputed[cols_onehot].fillna(0).astype(float)

    # --- 3️⃣ Standardization ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train_imputed.columns,
        index=X_train_imputed.index
    )
    X_pred_scaled = pd.DataFrame(
        scaler.transform(X_pred_imputed),
        columns=X_pred_imputed.columns,
        index=X_pred_imputed.index
    )

    # --- 4️⃣ Train the model ---
    model = regressor(**params)
    model.fit(X_train_scaled, y_train)

    # --- 5️⃣ Special case: XGBRegressor confidence ---
    if isinstance(model, XGBRegressor) and len(X_pred_scaled) > 0:
        booster = model.get_booster()
        n_trees = booster.num_boosted_rounds()
        dtest = xgb.DMatrix(X_pred_scaled)

        # Cumulative predictions
        cumulative_preds = np.array([
            booster.predict(dtest, iteration_range=(0, i))
            for i in range(1, n_trees + 1)
        ])
        # shape = (n_trees, n_samples)
        y_pred_mean = cumulative_preds[-1]
        y_pred_var = cumulative_preds.var(axis=0)

        df_conf = pd.DataFrame({'mean': y_pred_mean, 'var': y_pred_var})
        var_threshold = df_conf['var'].quantile(confidence_quantile)
        mask_conf = df_conf['var'] <= var_threshold
        n_selected = mask_conf.sum()

        # --- Visualization ---
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        sc = axs[0].scatter(df_conf['mean'], df_conf['var'], alpha=0.6, c=df_conf['var'], cmap='viridis')
        axs[0].axhline(var_threshold, color='red', linestyle='--')
        axs[0].set_title(f'{target_name} — Variance vs Prediction')
        axs[0].set_xlabel('Mean Prediction')
        axs[0].set_ylabel('Variance')
        plt.colorbar(sc, ax=axs[0])

        sns.histplot(df_conf['var'], bins=40, kde=True, ax=axs[1], color='skyblue')
        axs[1].axvline(var_threshold, color='red', linestyle='--', label=f'Quantile {confidence_quantile}')
        axs[1].legend()
        axs[1].set_title(f'Variance Distribution ({n_selected}/{len(df_conf)} kept)')
        plt.show()

        print(f"{target_name}: {n_selected}/{len(df_conf)} pseudo-labels kept (quantile={confidence_quantile}).")

        # Fill only confident predictions
        df.loc[X_pred_scaled.index[mask_conf], target_name] = y_pred_mean[mask_conf]

    else:
        # --- Other regressor (no confidence) ---
        if len(X_pred_scaled) > 0:
            y_pred = model.predict(X_pred_scaled)
            df.loc[mask_pred, target_name] = y_pred
            print(f"{target_name}: {n_to_fill} missing values filled (no confidence weighting).")
        else:
            print(f"{target_name}: no instances to predict after filtering.")

    return (df, model) if return_model else df
