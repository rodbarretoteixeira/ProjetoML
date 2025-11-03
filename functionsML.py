import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from collections import Counter
import math

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def fix_typos(col, db):
    """Fix edge typos, prioritizing matching first letters, then falling back."""
   
    db[col] = db[col].apply(lambda x: x.lower().strip().replace(r'\s+', ' ') if pd.notna(x) else x)
    unique_values = db[col].dropna().unique()
    corrected = db[col].copy()
    for full in unique_values:
        for candidate in unique_values:
            if len(candidate) + 2 == len(full) and full[1:-1] == candidate:
                corrected = corrected.replace(candidate, full)
                continue
            if len(candidate) + 1 == len(full) and full[1:] == candidate:
                corrected = corrected.replace(candidate, full)
                continue
            if len(candidate) + 1 == len(full) and full[:-1] == candidate:
                corrected = corrected.replace(candidate, full)
                continue

    db[col] = corrected

    return db


def fill_NaN_with_categorical(df, target_col, helper_cols):
    """
    Fill NaN values in target_col using mode within groups defined by helper_cols.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column to fill.
        helper_cols (list of str): List of 1 or 2 categorical helper columns.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled in target_col.
    """

    df = df.copy()
    
    # Fill missing helper columns with their mode
    for col in helper_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Group by helper columns (all rows included)
    df_filled = df.groupby(helper_cols, dropna=False, group_keys=False ).apply(fill_group_cat, target_col)
    
    return df_filled


def fill_NaN_with_mixed(df, target_col, cat_col, num_col, n_bins=15):
    """
    Fill NaN values in target_col using a combination of one categorical column
    and one numerical column (binned into ranges).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column with NaNs to fill.
        cat_col (str): Categorical helper column.
        num_col (str): Numerical helper column (to be binned).
        n_bins (int): Number of bins for numerical column.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled in target_col.
    """
    df = df.copy()

    # Bin numerical column and convert to string
    numeric_bins = pd.cut(df[num_col], bins=n_bins, duplicates='drop').astype(str)

    # Create combined key of categorical + numeric bin
    combined_key = df[cat_col].astype(str) + "_" + numeric_bins
    df['_combined_key'] = combined_key

    # Group by combined key and apply filling
    df_filled = df.groupby('_combined_key', group_keys=False).apply(fill_group_cat, target_col)

    # Drop the helper column
    df_filled = df_filled.drop(columns=['_combined_key'])

    return df_filled

def fill_group_cat(group, target_col):
    mode_value = group[target_col].mode()
    if not mode_value.empty:
        group[target_col] = group[target_col].fillna(mode_value[0])
    return group


def fill_NaN_with_numeric(df, target_col, helper_cols):
    """
    Fill NaN values in a numerical column using 1 or 2 helper columns to compute group-wise statistic.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column with NaNs to fill.
        helper_cols (list of str): List of 1 or 2 helper columns.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled in target_col.
    """
    df = df.copy()  
    # Apply group-wise filling
    df[target_col] = df[target_col].fillna(df.groupby(helper_cols)[target_col].transform('median'))
    
    return df


def missing_values_table(df):
    """
    Prints a table showing the number and percentage of missing values
    for each column in the DataFrame.
    """
    # Calculate count and percentage of NaNs per column
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    # Combine into a clean DataFrame
    missing_df = pd.DataFrame({
        'Missing Values': missing_count,
        'Percent Missing (%)': missing_percent.round(2)
    })

    # Filter out columns with no missing values and sort descending
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(
        by='Percent Missing (%)', ascending=False
    )

    # Print results
    print(f"Total columns: {df.shape[1]}")
    print(f"Columns with missing values: {missing_df.shape[0]}\n")

    return missing_df

def negative_values_table(df):
    """
    Prints a table showing the number and percentage of negative numeric values
    for each column in the DataFrame.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    
    # Count negatives per column
    negative_count = (numeric_df < 0).sum()
    negative_percent = (negative_count / len(numeric_df)) * 100

    # Combine into a clean DataFrame
    negative_df = pd.DataFrame({
        'Negative Values': negative_count,
        'Percent Negative (%)': negative_percent.round(2)
    })

    # Filter out columns with no negative values and sort descending
    negative_df = negative_df[negative_df['Negative Values'] > 0].sort_values(
        by='Percent Negative (%)', ascending=False
    )

    # Print summary
    print(f"Total numeric columns: {numeric_df.shape[1]}")
    print(f"Columns with negative values: {negative_df.shape[0]}\n")

    return negative_df


def irrational_values_table(df, decimal_threshold=3):
    """
    Displays how many 'irrational' (overly precise float) numbers exist per column,
    plus a total count of rows that contain at least one such value.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    total_rows = len(df)

    def is_irrational(s):
        dec_len = s.astype(str).str.extract(r'\.(\d+)')[0].str.len()
        dec_len = pd.to_numeric(dec_len, errors="coerce").fillna(0).astype(int)
        return dec_len > decimal_threshold

    irrational_mask = df[num_cols].apply(is_irrational)
    counts = irrational_mask.sum()

    irrational_df = pd.DataFrame({
        "Irrational Count": counts,
        "Percent Irrational (%)": (counts / total_rows * 100).round(2)
    }).query("`Irrational Count` > 0").sort_values("Percent Irrational (%)", ascending=False)

    total_rows_irrational = irrational_mask.any(axis=1).sum()
    total_percent = (total_rows_irrational / total_rows * 100).round(2)

    total_row = pd.DataFrame({
        "Irrational Count": [total_rows_irrational],
        "Percent Irrational (%)": [total_percent]
    }, index=["Total (rows with any irrational value)"])

    print(f"Total numeric columns: {len(num_cols)}")
    print(f"Columns with irrational values: {irrational_df.shape[0]}")
    print(f"Rows with at least one irrational value: {total_rows_irrational}\n")

    return pd.concat([irrational_df, total_row])





def negative_to_nan_columns(columns, df):
    """Convert negative numeric values to NaN for specified columns."""
    df.loc[:, columns] = df[columns].where(df[columns] >= 0)
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

def conditional_entropy(x, y):
    """Compute the conditional entropy H(X | Y)."""
    # x, y: array-like, same length
    y_counter = Counter(y)
    xy_counter = Counter(zip(x, y))
    total = len(x)
    ent = 0.0
    for (x_val, y_val), joint_count in xy_counter.items():
        p_xy = joint_count / total
        p_y = y_counter[y_val] / total
        ent += p_xy * math.log(p_y / p_xy, 2)
    return ent

def theils_u(x, y):
    """
    Compute Theil’s U (Uncertainty Coefficient) U(X|Y):
    how much knowing Y reduces uncertainty in X.
    Returns 0 ≤ U ≤ 1. Asymmetric: U(X|Y) ≠ U(Y|X).
    """
    # Drop missing simultaneously
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return np.nan

    x = df["x"]
    y = df["y"]

    # Entropy of X
    x_counter = Counter(x)
    total = len(x)
    p_x = [cnt / total for cnt in x_counter.values()]
    # If entropy of X is zero (i.e. X is constant), we define U = 1
    # since Y gives "complete information" (but there is no uncertainty in X anyway)
    s_x = 0.0
    for p in p_x:
        s_x -= p * math.log(p, 2)

    if s_x == 0:
        return 1.0

    # Conditional entropy H(X|Y)
    s_x_given_y = conditional_entropy(x, y)

    # U(X|Y) = (H(X) – H(X|Y)) / H(X)
    return (s_x - s_x_given_y) / s_x



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


def TestCorrelationRatio(categories, values, var, threshold=0.1):
    """
    Test the strength of association between a categorical and a numeric variable
    using the Correlation Ratio (η).

    Parameters
    ----------
    categories : array-like
        Categorical variable (predictor).
    values : array-like
        Numeric variable (target).
    var : str
        Name of the categorical variable (for display).
    threshold : float, optional
        Minimum η value considered "important" (default = 0.1).

    Prints
    ------
    A message indicating whether the variable is important for prediction.
    Returns
    -------
    float
        The computed correlation ratio η.
    """
    df = pd.DataFrame({'cat': categories, 'val': pd.to_numeric(values, errors='coerce')}).dropna()
    if df.empty:
        print(f"{var}: no valid data available.")
        return np.nan

    means = df.groupby('cat')['val'].mean()
    overall_mean = df['val'].mean()
    n_per_cat = df.groupby('cat').size()

    numerator = (n_per_cat * (means - overall_mean) ** 2).sum()
    denominator = ((df['val'] - overall_mean) ** 2).sum()

    if denominator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)

    # Print a simple, interpretable message
    if eta >= threshold:
        print(f"{var} is IMPORTANT for prediction (η = {eta:.3f})")
    else:
        print(f"{var} is NOT an important predictor (η = {eta:.3f}, below {threshold})")

    return eta


def plot_multiple_boxes_with_outliers(data, columns, ncols=2):

    num_columns = len(columns)
    nrows = (num_columns + ncols - 1) // ncols
    plt.figure(figsize=(8 * ncols, 4 * nrows))

    for i, column in enumerate(columns):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]

        plt.subplot(nrows, ncols, i + 1)
        plt.boxplot(data[column], vert=False, widths=0.7,
                    patch_artist=True, boxprops=dict(facecolor='#4CAF50', color='black'),
                    medianprops=dict(color='black'))
        plt.scatter(outliers, [1] * len(outliers), color='red', marker='o', label='Outliers')
        plt.title(f"Box Plot of {column} with Outliers")
        plt.xlabel('Value')
        plt.yticks([])
        plt.legend()

    plt.tight_layout()
    plt.show()



def IQR_outliers(df, variables):
    q1 = df[variables].quantile(0.25)
    q3 = df[variables].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    for col in variables:
        outliers = df[(df[col] < lower[col]) | (df[col] > upper[col])]
        print(f"{col}: {len(outliers)} outliers")

    return None  # não devolve o DataFrame inteiro


def kfold_target_encode(train_df, cat_cols, target_col, id_col='carID', n_splits=5):
    """
    Perform K-Fold Target Encoding for one or more categorical columns combined.

    Parameters:
        train_df (pd.DataFrame): Input DataFrame
        cat_cols (str or list): Categorical column(s) to encode (e.g. ['brand', 'model'])
        target_col (str): Target column (e.g., 'price')
        id_col (str): Unique identifier column (e.g., 'carID')
        n_splits (int): Number of folds

    Returns:
        pd.DataFrame: DataFrame with a new encoded column based on combined categories
    """

    # Ensure ID column is unique
    if not train_df[id_col].is_unique:
        raise ValueError(f"{id_col} must contain unique values for KFold target encoding.")

    # Create a copy to avoid modifying the original
    df = train_df.copy()

    # Allow single string or list for cat_cols
    if isinstance(cat_cols, str):
        cat_cols = [cat_cols]

    # Create combined categorical feature in a temporary variable
    combo_series = df[cat_cols].astype(str).agg('_'.join, axis=1)
    combo_name = "_".join(cat_cols)
    encoded_col = f"{combo_name}_encoded"

    # Initialize encoded column
    df[encoded_col] = np.nan

    # Create the KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform out-of-fold target encoding
    for train_idx, valid_idx in kf.split(df):
        train_fold = df.iloc[train_idx].copy()
        valid_fold = df.iloc[valid_idx].copy()

        # Create temporary combined category columns for this fold
        train_combo = combo_series.iloc[train_idx]
        valid_combo = combo_series.iloc[valid_idx]

        # Compute mean target per combined category on training fold
        fold_mean = train_fold.groupby(train_combo)[target_col].mean()

        # Map encoded values for the validation fold
        df.loc[df.index[valid_idx], encoded_col] = valid_combo.map(fold_mean).values

    # Fill unseen categories with overall mean
    overall_mean = df[target_col].mean()
    df[encoded_col].fillna(overall_mean, inplace=True)

    # Return only original columns + new encoded column
    return df[[*train_df.columns, encoded_col]]


