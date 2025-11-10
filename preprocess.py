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

def replace_N_with_NaN(df):
    """
    Replace all string values equal to 'N' with NaN in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all occurrences of 'N' replaced by np.nan.
    """
    return df.replace("N", np.nan)


def identify_columns(df):
    """
    Assign proper column names to the input DataFrame and remove the 'Weld ID' column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame without named columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns and the 'Weld ID' column removed.
    """
    columns = [
        "Carbon concentration", "Silicon concentration", "Manganese concentration",
        "Sulphur concentration", "Phosphorus concentration", "Nickel concentration",
        "Chromium concentration", "Molybdenum concentration", "Vanadium concentration",
        "Copper concentration", "Cobalt concentration", "Tungsten concentration",
        "Oxygen concentration", "Titanium concentration", "Nitrogen concentration",
        "Aluminium concentration", "Boron concentration", "Niobium concentration",
        "Tin concentration", "Arsenic concentration", "Antimony concentration",
        "Current", "Voltage", "AC or DC", "Electrode polarity", "Heat input",
        "Interpass temperature", "Type of weld",
        "Post weld heat treatment temperature", "Post weld heat treatment time",
        "Yield strength", "Ultimate tensile strength", "Elongation",
        "Reduction of area", "Charpy temperature", "Charpy impact toughness",
        "Hardness", "50% FATT", "Primary ferrite in microstructure",
        "Ferrite with second phase", "Acicular ferrite", "Martensite",
        "Ferrite with carbide aggregate", "Weld ID"
    ]

    df.columns = columns
    df = df.drop(columns=["Weld ID"], errors="ignore")

    return df



def process_value(value, ineq_strategy="half"):
    """
    Process numeric values and inequalities represented as strings.
    
    Parameters
    ----------
    value : any
        The value to process (can be a string like '<50' or a number).
    ineq_strategy : str, default="half"
        Strategy to handle inequality values starting with '<':
        - "half"   : replace "<x" with x / 2
        - "minus10": replace "<x" with x - 10% (x * 0.9)
        - "zero"   : replace "<x" with 0
        - any other : return x as-is

    Returns
    -------
    float or np.nan
        The processed numeric value, or NaN if conversion fails.
    """
    value = str(value).strip()

    if value.startswith("<"):
        try:
            x = float(value[1:])
            if ineq_strategy == "half":
                return x / 2
            elif ineq_strategy == "minus10":
                return x * 0.9
            elif ineq_strategy == "zero":
                return 0.0
            else:
                return x
        except ValueError:
            return np.nan
    else:
        try:
            return float(value)
        except ValueError:
            return np.nan


def convert_numeric_columns(df, skip_indices=None, extract_indices=None, ineq_strategy="half"):
    """
    Convert numeric-like columns in a DataFrame and handle inequality values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.
    skip_indices : list of int, optional
        Column indices to skip during conversion.
    extract_indices : list of int, optional
        Columns where numeric extraction (e.g., from '150-200') should be applied.
    ineq_strategy : str, default="half"
        Strategy to handle inequality values (see `process_value`).

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with converted numeric columns.
    """
    if skip_indices is None:
        skip_indices = [23, 24, 27, 43]
    if extract_indices is None:
        extract_indices = [14, 26, 36]

    df = df.copy()

    for i, col in enumerate(df.columns):
        if i in skip_indices:
            continue

        # Handle columns requiring numeric extraction
        if i in extract_indices:
            df[col] = (
                df[col].astype(str)
                .str.extract(r"^(\d+)", expand=False)
                .replace('150-200', 175)
            )

        # Apply numeric conversion with inequality handling
        df[col] = df[col].apply(lambda v: process_value(v, ineq_strategy))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def encode_features(df):
    """
    Perform one-hot encoding on categorical features and handle missing values by adding an 'Unknown' category.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical and numerical features.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables replaced by their one-hot encoded representations.
        Missing values are treated as a separate "Unknown" category.
    """
    # Select categorical columns
    cat_df = df.select_dtypes(include=['object'])
    
    # Replace NaN with "Unknown" to preserve missing category information
    cat_df = cat_df.fillna("Unknown")
    
    # Apply one-hot encoding (do not drop the first column to keep all categories)
    df_encoded = pd.get_dummies(cat_df, drop_first=False).astype(int)
    
    # Concatenate numerical columns and encoded categorical columns
    df = pd.concat([df.select_dtypes(exclude=['object']), df_encoded], axis=1)
    
    return df


def handle_value(val, ineq_strategy="half"):
    """
    Process numerical values and inequality strings.

    Parameters
    ----------
    val : any
        The value to process (can be a string or numeric).
    ineq_strategy : str, default "half"
        Strategy for handling inequality values (strings starting with '<'):
        - "half"    : replace "<x" with x / 2
        - "minus10" : replace "<x" with x - 10% (i.e., x * 0.9)
        - "zero"    : replace "<x" with 0
        - any other : return x unchanged

    Returns
    -------
    float or np.nan
        Processed numeric value or NaN if conversion fails.
    """
    val = str(val).strip()
    
    if val.startswith("<"):
        try:
            x = float(val[1:])
            if ineq_strategy == "half":
                return x / 2
            elif ineq_strategy == "minus10":
                return x * 0.9
            elif ineq_strategy == "zero":
                return 0.0
            else:
                return x
        except ValueError:
            return np.nan
    else:
        try:
            return float(val)
        except ValueError:
            return np.nan


def convert_values(df, to_avoid=None, replace_spec=None, ineq_strategy="half"):
    """
    Convert numerical columns to the correct format and handle inequality or range values
    according to the specified strategy.

    Parameters:
    -------------
    df : pd.DataFrame
        Input DataFrame with numerical columns.
    to_avoid : list of int, optional
        List of column indices to skip (do not convert).
    replace_spec : list of int, optional
        List of column indices where special replacements are applied (e.g., extracting numbers from ranges).
    ineq_strategy : str, default "half"
        Strategy to handle inequality values:
        - "half" : replace '<x' by x/2
        - "zero" : replace '<x' by 0
        - "minus10" : replace '<x' by x*0.9
    """
    if to_avoid is None:
        to_avoid = [23, 24, 27, 43]
    if replace_spec is None:
        replace_spec = [14, 26, 36]

    for i, col in enumerate(df.columns):
        if i not in to_avoid:
            if i in replace_spec:
                # Extract numerical values from the column (e.g., remove non-numeric characters)
                df[col] = df[col].astype(str).str.extract(r'^(\d+)')

                # Replace a specific range string with its mean value
                df[col] = df[col].replace('150-200', 175)
            
            # Apply the inequality strategy to handle values like '<x'
            df[col] = df[col].apply(lambda v: handle_value(v, ineq_strategy))

            # Convert the column to numeric, setting errors to NaN if conversion fails
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df






def separate_concentration_features(X_train, X_test, target_cols):
    """
    Separate concentration-related columns from other feature columns in training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Test feature set.
    target_cols : list
        List of target variable column names to exclude from feature sets.

    Returns
    -------
    X_train_concentration : pd.DataFrame
        Training DataFrame containing only columns ending with 'concentration'.
    X_train_rest : pd.DataFrame
        Training DataFrame containing all other (non-concentration and non-target) features.
    X_test_concentration : pd.DataFrame
        Test DataFrame containing only columns ending with 'concentration'.
    X_test_rest : pd.DataFrame
        Test DataFrame containing all other (non-concentration and non-target) features.
    """
    # Identify columns related to concentration
    concentration_cols = [col for col in X_train.columns if col.endswith("concentration")]

    # Separate concentration features from the rest for training data
    X_train_concentration = X_train.loc[:, concentration_cols]
    X_train_rest = X_train.loc[:, [col for col in X_train.columns if col not in target_cols + concentration_cols]]

    # Do the same for test data
    X_test_concentration = X_test.loc[:, concentration_cols]
    X_test_rest = X_test.loc[:, [col for col in X_test.columns if col not in target_cols + concentration_cols]]

    return X_train_concentration, X_train_rest, X_test_concentration, X_test_rest



def fill_concentration_values_train_test(X_train_conc, X_test_conc):
    """
    Impute missing values in concentration features separately for the training and test sets.
    The imputation model is fitted on the training set and applied to the test set 
    to prevent data leakage.

    Strategy:
    ----------
    - 'Phosphorus concentration' and 'Sulphur concentration' are imputed using IterativeImputer 
      with a mean-based strategy.
    - All other concentration columns are filled with zeros.

    Parameters
    ----------
    X_train_conc : pd.DataFrame
        Training set containing concentration-related features.
    X_test_conc : pd.DataFrame
        Test set containing concentration-related features.

    Returns
    -------
    X_train_filled : pd.DataFrame
        Training set with missing values imputed.
    X_test_filled : pd.DataFrame
        Test set with missing values imputed.
    """

    # Columns to be imputed using mean strategy
    cols_mean = [c for c in ['Sulphur concentration', 'Phosphorus concentration'] if c in X_train_conc.columns]
    # Columns to be filled with zero
    cols_zero = [c for c in X_train_conc.columns if c not in cols_mean]

    # --- Training set imputation ---
    train_parts = []

    if cols_mean:
        imp_mean = IterativeImputer(initial_strategy="mean", sample_posterior=True, random_state=42)
        train_mean_filled = pd.DataFrame(
            imp_mean.fit_transform(X_train_conc[cols_mean]),
            columns=cols_mean,
            index=X_train_conc.index
        )
        train_parts.append(train_mean_filled)

    if cols_zero:
        train_zero_filled = X_train_conc[cols_zero].fillna(0)
        train_parts.append(train_zero_filled)

    X_train_filled = pd.concat(train_parts, axis=1)

    # --- Test set imputation ---
    test_parts = []

    if cols_mean:
        test_mean_filled = pd.DataFrame(
            imp_mean.transform(X_test_conc[cols_mean]),
            columns=cols_mean,
            index=X_test_conc.index
        )
        test_parts.append(test_mean_filled)

    if cols_zero:
        test_zero_filled = X_test_conc[cols_zero].fillna(0)
        test_parts.append(test_zero_filled)

    X_test_filled = pd.concat(test_parts, axis=1)

    # Reorder columns to match the original DataFrame
    X_train_filled = X_train_filled[X_train_conc.columns]
    X_test_filled = X_test_filled[X_test_conc.columns]

    return X_train_filled, X_test_filled


def fill_remaining_features_train_test(X_train_rest, X_test_rest):
    """
    Impute missing values in non-concentration features using IterativeImputer.
    The imputer is fitted on the training set and applied to the test set 
    to avoid data leakage.

    Parameters
    ----------
    X_train_rest : pd.DataFrame
        Training subset containing non-concentration features.
    X_test_rest : pd.DataFrame
        Test subset containing non-concentration features.

    Returns
    -------
    X_train_rest_filled : pd.DataFrame
        Training set with imputed values.
    X_test_rest_filled : pd.DataFrame
        Test set with imputed values.
    """

    # Initialize imputer
    imp_rest = IterativeImputer(initial_strategy="mean", sample_posterior=True, random_state=42)

    # --- Fit on train, transform both ---
    X_train_rest_filled = pd.DataFrame(
        imp_rest.fit_transform(X_train_rest),
        columns=X_train_rest.columns,
        index=X_train_rest.index
    )

    X_test_rest_filled = pd.DataFrame(
        imp_rest.transform(X_test_rest),
        columns=X_test_rest.columns,
        index=X_test_rest.index
    )

    return X_train_rest_filled, X_test_rest_filled


def impute_missing_values(X_train, X_test, target_cols):
    """
    Main preprocessing function for imputing missing values.

    Steps:
    -------
    Split features into concentration-related and non-concentration subsets.
    Apply appropriate imputation strategies:
       - Concentration columns: specialized imputation (zero or mean-based).
       - Other columns: iterative imputation.
    Recombine the processed subsets into complete train and test DataFrames.
    Preserve original column order for consistency.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set of features.
    X_test : pd.DataFrame
        Test set of features.
    target_cols : list
        Names of target columns to exclude from feature imputation.

    Returns
    -------
    X_train_processed : pd.DataFrame
        Training features after all imputations.
    X_test_processed : pd.DataFrame
        Test features after all imputations.
    """

    # Separate concentration-related and non-concentration features
    X_train_conc, X_train_rest, X_test_conc, X_test_rest = separate_concentration_features(
        X_train, X_test, target_cols
    )

    # Impute concentration-related columns
    X_train_conc_filled, X_test_conc_filled = fill_concentration_values_train_test(
        X_train_conc, X_test_conc
    )

    # Impute other feature columns
    X_train_rest_filled, X_test_rest_filled = fill_remaining_features_train_test(
        X_train_rest, X_test_rest
    )

    # Recombine both subsets
    X_train_processed = pd.concat([X_train_conc_filled, X_train_rest_filled], axis=1)
    X_test_processed = pd.concat([X_test_conc_filled, X_test_rest_filled], axis=1)

    # Preserve original column order for consistency
    X_train_processed = X_train_processed[X_train.columns]
    X_test_processed = X_test_processed[X_test.columns]

    return X_train_processed, X_test_processed



def preprocess_data(df_origin,ineq_strategy):
    df=df_origin.copy()
    df=replace_N_with_NaN(df)
    df=identify_columns(df)
    df=convert_numeric_columns(df, skip_indices=None, extract_indices=None, ineq_strategy=ineq_strategy)
    df=encode_features(df)
    return(df)