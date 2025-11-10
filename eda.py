import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_analysis_2(df,
                  missingness=False,
                  basic_info=False,
                  feature_distrib=False,
                  categorical_distrib=False,
                  correlation_matrix=False,
                  high_correlation=False,
                  outlier_detection=False,
                  pair_analysis=False,
                  target=None):
    """
    Notebook-friendly exploratory data analysis (EDA) with True/False flags for sections.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    missingness : bool
        Display missing value table.
    basic_info : bool
        Display dataset overview and descriptive statistics.
    feature_distrib : bool
        Display summary of numeric features.
    categorical_distrib : bool
        Display summary of categorical features.
    correlation_matrix : bool
        Display correlation matrix for numeric features.
    high_correlation : bool
        Display highly correlated numeric pairs.
    outlier_detection : bool
        Display potential outlier counts per numeric feature.
    pair_analysis : bool
        Display correlations between all numeric feature pairs.
    target : str or None
        Target column to analyze numeric correlations with.
    """
    
    # Basic info
    if basic_info:
        print("### Dataset Overview")
        print(f"- Number of observations: {df.shape[0]}")
        print(f"- Number of features: {df.shape[1]}")
        
        overview = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notna().sum(),
            "Missing %": df.isna().mean() * 100,
            "Dtype": df.dtypes
        }).sort_values("Missing %", ascending=False)
        numeric_cols = overview.select_dtypes(include=np.number).columns
        display(
            overview.style
            .background_gradient(cmap="Blues", subset=numeric_cols)
            .format({col: "{:.2f}" for col in numeric_cols})
        )
        
        # Descriptive stats for numeric features
        print("\n### Descriptive Statistics")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            display(df[numeric_cols].describe().style.background_gradient(cmap="Greens").format("{:.2f}"))
    
    # Missing values
    if missingness:
        print("\n### Missing Values Table")
        missing_data = pd.DataFrame({
            "unique_values": df.nunique(),
            "missing_rate": df.isna().mean() * 100
        }).sort_values("missing_rate", ascending=False)
        display(missing_data.style.background_gradient(cmap="Reds").format("{:.2f}"))
    
    # Numeric distributions
    if feature_distrib:
        print("\n### Numeric Feature Distributions")
        numeric_cols = df.select_dtypes(include="number").columns
        n_cols = 4
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        plt.figure(figsize=(20, 4 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[col], kde=True, bins=30, color='steelblue')
            plt.title(f"Distribution of {col}")
            plt.xlabel("")
            plt.ylabel("")
        plt.tight_layout()
        plt.show()

    # Categorical distributions
    if categorical_distrib:
        print("\n### Categorical Feature Distributions (Top 20 values)")
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            n_cols = 3
            n_rows = int(np.ceil(len(cat_cols) / n_cols))
            plt.figure(figsize=(18, 4 * n_rows))
            for i, col in enumerate(cat_cols, 1):
                plt.subplot(n_rows, n_cols, i)
                df[col].value_counts().head(20).plot(kind='bar', color='orange')
                plt.title(f"{col}")
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    # Correlation matrix
    numeric_cols = df.select_dtypes(include="number").columns
    if correlation_matrix and len(numeric_cols) > 1:
        print("\n### Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(25,20))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlations")
        plt.show()

    # Highly correlated pairs
    if high_correlation and len(numeric_cols) > 1:
        # compute correlation on numeric columns only
        corr_matrix = df[numeric_cols].corr().abs()

        # get upper triangle
        high_corr_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

        # ensure correlation column is numeric
        high_corr_pairs['Correlation'] = pd.to_numeric(high_corr_pairs['Correlation'], errors='coerce')

        # filter highly correlated
        high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] > 0.8].sort_values("Correlation", ascending=False)

        if not high_corr_pairs.empty:
            # Ensure Correlation column is numeric
            high_corr_pairs['Correlation'] = pd.to_numeric(high_corr_pairs['Correlation'], errors='coerce')
            
            print("\n### Highly Correlated Feature Pairs (|corr| > 0.8)")
            
            # Apply style only to numeric columns
            numeric_cols = high_corr_pairs.select_dtypes(include='number').columns
            display(high_corr_pairs.style.background_gradient(cmap="coolwarm", subset=numeric_cols)
                                    .format("{:.2f}", subset=numeric_cols))


    # Outlier detection
    if outlier_detection and len(numeric_cols) > 0:
        print("\n### Potential Outliers")
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            print(f"{col}: {n_outliers} potential outliers")
    
    # Pairwise analysis
    if pair_analysis and len(numeric_cols) > 1:
        print("\n### Pairwise Correlations")
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = df[col1].corr(df[col2])
                print(f"{col1} vs {col2}: {corr:.3f}")
    
    # Target variable analysis
    if target is not None and target in numeric_cols:
        print(f"\n### Target Variable Analysis: {target}")
        target_corr = df[numeric_cols].corr()[target].drop(target).sort_values(key=abs, ascending=False)
        display(target_corr.to_frame(name="Correlation").style.background_gradient(cmap="coolwarm").format("{:.2f}"))
    
    print("\n End of report")
