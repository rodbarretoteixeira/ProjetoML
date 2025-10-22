import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def fix_typos(col, db):
    """Fix edge typos, prioritizing matching first letters, then falling back."""
   
    db[col] = db[col].apply(lambda x: x.lower().strip().replace(r'\s+', ' ') if pd.notna(x) else x)
    unique_col = sorted(db[col].dropna().unique())
    replacements = {}

    for i, u1 in enumerate(unique_col):
        for u2 in unique_col[i+1:]:
            if u1 == u2:
                continue

            # Extract middle cores
            u1_core = u1[1:-1] if len(u1) > 2 else u1
            u2_core = u2[1:-1] if len(u2) > 2 else u2

            # Only consider merges if first letters match
            if u1[0] == u2[0]:

                # Case 1: middle match → pick the longer one
                if u1_core == u2_core:
                    bigger = u1 if len(u1) > len(u2) else u2
                    smaller = u2 if bigger == u1 else u1
                    replacements[smaller] = bigger
                    continue

                # Case 2: last-letter-only typos
                if u1.startswith(u2) or u2.startswith(u1):
                    bigger = u1 if len(u1) > len(u2) else u2
                    smaller = u2 if bigger == u1 else u1
                    replacements[smaller] = bigger
                    continue

    # Apply replacements twice for propagation
    db[col] = db[col].replace(replacements)
    db[col] = db[col].replace(replacements)

    # --- Fix entries missing the first letter ---
    def fix_missing_first_letter(column):
        unique_values = column.dropna().unique()
        corrected = column.copy()
        for full in unique_values:
            for candidate in unique_values:
                # Only fix shorter candidates that are missing the first letter
                if len(candidate) + 1 == len(full) and full[1:] == candidate:
                    corrected = corrected.replace(candidate, full)
        return corrected

    db[col] = fix_missing_first_letter(db[col])

    return db


def fill_invalid_by_category(target_col, df, category_col):
    """
    Fills NaN values in `target_col` based on `category_col`.
    - Numeric columns: mean per category
    - Categorical columns: mode per category
    Handles Int64, float, and object types safely.
    """
    mask = df[target_col].isna()
    orig_dtype = df[target_col].dtype

    if pd.api.types.is_numeric_dtype(df[target_col]):
        # Ensure numeric columns are float during filling
        df[target_col] = df[target_col].astype(float)
        agg_values = df.groupby(category_col)[target_col].mean()
    else:
        agg_values = df.groupby(category_col)[target_col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )

    df.loc[mask, target_col] = df.loc[mask, category_col].map(agg_values)

    # If originally Int64, round floats before casting
    if str(orig_dtype) == "Int64":
        df[target_col] = df[target_col].round().astype("Int64")
    else:
        try:
            df[target_col] = df[target_col].astype(orig_dtype)
        except Exception:
            pass

    return df


def cramers_v(x, y):
    """
    Compute Cramér's V statistic for measuring association between two categorical variables.
    
    Parameters:
        x (array-like): First categorical variable
        y (array-like): Second categorical variable
    
    Returns:
        float: Cramér's V value (0 = no association, 1 = perfect association)
    """
    # Build the contingency table
    ct = pd.crosstab(x, y)
    
    # Compute Chi-squared statistic
    chi2 = chi2_contingency(ct)[0]
    
    # Total number of observations
    n = ct.sum().sum()
    
    # Minimum dimension - 1
    min_dim = min(ct.shape) - 1
    
    if min_dim == 0:
        return 0.0  # Avoid division by zero
    
    return np.sqrt(chi2 / (n * min_dim))

def correlation_ratio(categories, values):
    """
    Compute the Correlation Ratio (η) between a categorical and a numeric variable.

    Parameters:
        categories (array-like): Categorical variable
        values (array-like): Numeric variable

    Returns:
        float: Correlation Ratio (η), between 0 (no relationship) and 1 (perfect)
    """
    # Drop missing pairs
    df = pd.DataFrame({'cat': categories, 'val': values}).dropna()
    if df.empty:
        return np.nan

    # Mean per category
    means = df.groupby('cat')['val'].mean()
    overall_mean = df['val'].mean()

    # Between-group and total variance
    n_per_cat = df.groupby('cat').size()
    numerator = (n_per_cat * (means - overall_mean) ** 2).sum()
    denominator = ((df['val'] - overall_mean) ** 2).sum()

    if denominator == 0:
        return 0.0

    return np.sqrt(numerator / denominator)

def plot_histogram(data, xlabel, ylabel, title, color='green'):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be plotted.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the histogram.
        color (str): The color of the histogram bars. Default is 'green'.
    """
    plt.hist(data, edgecolor='black', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()