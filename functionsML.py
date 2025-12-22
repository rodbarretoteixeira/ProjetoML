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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import HistGradientBoostingRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor

import random
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

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


# def fill_group_cat(group, target_col, train_grouped):
#     """
#     Fill NaNs in the target column for a group using the mode
#     from the corresponding group in train_grouped.
#     """
#     try:
#         # Get the corresponding group from train_grouped
#         key = group.name
#         train_group = train_grouped.get_group(key)
#         mode_value = train_group[target_col].mode()
#         if not mode_value.empty:
#             group[target_col] = group[target_col].fillna(mode_value[0])
#     except KeyError:
#         # If no corresponding group in train, fill with overall mode
#         mode_value = train_grouped.obj[target_col].mode()
#         if not mode_value.empty:
#             group[target_col] = group[target_col].fillna(mode_value[0])
#     return group



# def fill_NaN_with_categorical(df, train_db, target_col, helper_cols):
#     """Fills NaNs using training-based modes to avoid leakage."""
#     df = df.copy()
    
#     # 1. Create a specific lookup from TRAIN only
#     # We use .agg(list).map(lambda x: mode) to handle multiple modes safely
#     lookup = train_db.groupby(helper_cols)[target_col].apply(
#         lambda x: x.mode()[0] if not x.mode().empty else None
#     ).to_dict()

#     # 2. Map the lookup. Since helper_cols is a list, we create a temporary key
#     def get_val(row):
#         key = tuple(row[col] for col in helper_cols)
#         return lookup.get(key, None)

#     df[target_col] = df[target_col].fillna(df.apply(get_val, axis=1))

#     # 3. Fallback to global mode from TRAIN
#     global_mode = train_db[target_col].mode()[0]
#     df[target_col] = df[target_col].fillna(global_mode)
#     return df

# def fill_NaN_with_mixed(df, train_db, target_col, cat_col, num_col, n_bins=30):
#     """Fills NaNs using shared bin edges to ensure consistency."""
#     df = df.copy()
    
#     # FIX: Calculate bin edges ONLY on train_db
#     _, bins = pd.qcut(train_db[num_col], q=n_bins, retbins=True, duplicates='drop')
    
#     # Apply those EXACT bins to both
#     train_bins = pd.cut(train_db[num_col], bins=bins, labels=False, include_lowest=True)
#     df_bins = pd.cut(df[num_col], bins=bins, labels=False, include_lowest=True)

#     # Create lookup from binned train
#     lookup = train_db.assign(tmp_bin=train_bins).groupby([cat_col, 'tmp_bin'])[target_col].apply(
#         lambda x: x.mode()[0] if not x.mode().empty else None
#     ).to_dict()

#     # Map to df
#     def get_val_mixed(row, bin_val):
#         return lookup.get((row[cat_col], bin_val), None)

#     # Use a helper series to map
#     df_fill = pd.Series([lookup.get((c, b), None) for c, b in zip(df[cat_col], df_bins)], index=df.index)
#     df[target_col] = df[target_col].fillna(df_fill)

#     # Fallback to global mode from TRAIN
#     df[target_col] = df[target_col].fillna(train_db[target_col].mode()[0])
#     return df

def fill_NaN_with_categorical(df, target_col, helper_cols, binned=None):
    """
    Fill NaN values in target_col using mode within groups defined by helper_cols
    and optionally binned numeric columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Target column to fill NaNs.
        helper_cols (list of str): Categorical helper column names.
        binned (list of pd.Series, optional): List of binned numeric Series to include as helpers.

    Returns:
        pd.DataFrame: DataFrame with NaNs filled in target_col.
    """
    df = df.copy()

    # Initialize list of all helpers
    all_helpers = list(helper_cols)  # copy original strings

    # Add binned Series as temporary columns
    temp_cols = []
    if binned:
        for i, s in enumerate(binned):
            temp_name = f"_temp_binned_{i}"
            df[temp_name] = s
            all_helpers.append(temp_name)
            temp_cols.append(temp_name)

    # Fill missing values in all helper columns with mode
    for col in all_helpers:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Group by helpers and fill target column
    df_filled = df.groupby(all_helpers, dropna=False, group_keys=False).apply(fill_group_cat, target_col)

    # Remove temporary binned columns
    if temp_cols:
        df_filled = df_filled.drop(columns=temp_cols)

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


# def fill_NaN_with_mixed(df, target_col, cat_cols, num_cols, n_bins=30):
#     """
#     Fill NaN values in target_col using a combination of categorical and numerical columns.
#     Ignores helper columns with missing values for each row.

#     Parameters:
#         df (pd.DataFrame): Input DataFrame.
#         target_col (str): Column with NaNs to fill.
#         cat_cols (list of str): Categorical helper columns.
#         num_cols (list of str): Numerical helper columns (to be binned).
#         n_bins (int): Number of bins for numerical columns.

#     Returns:
#         pd.DataFrame: DataFrame with NaNs filled in target_col.
#     """
#     df = df.copy()

#     # Bin numerical columns
#     binned_num = pd.DataFrame(index=df.index)
#     for col in num_cols:
#         binned_num[col] = pd.cut(df[col], bins=n_bins, duplicates='drop').astype(str)

#     # Convert categorical columns to string
#     cat_df = df[cat_cols].astype(str) if cat_cols else pd.DataFrame(index=df.index)

#     # Combine all helper columns
#     helper_df = pd.concat([cat_df, binned_num], axis=1)

#     # Function to build key for each row, ignoring missing helper values
#     def build_key(row):
#         valid_values = row.dropna().astype(str)
#         return "_".join(valid_values) if not valid_values.empty else "ALL_NAN"

#     # Create combined key row-wise
#     df['_combined_key'] = helper_df.apply(build_key, axis=1)

#     # Group by combined key and fill NaNs
#     df_filled = df.groupby('_combined_key', group_keys=False).apply(lambda g: fill_group_cat(g, target_col))

#     # Drop helper column
#     df_filled = df_filled.drop(columns=['_combined_key'])
#     return df_filled

def fill_group_cat(group, target_col):
    mode_value = group[target_col].mode()
    if not mode_value.empty:
        group[target_col] = group[target_col].fillna(mode_value[0])
    return group


# def fill_group_cat(group, target_col):
#     mode_value = group[target_col].mode()
#     if not mode_value.empty:
#         group[target_col] = group[target_col].fillna(mode_value[0])
#     return group

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


import pandas as pd
import numpy as np
import random
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def check_overfitting(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """
    Analyzes Overfitting by comparing Train vs Validation metrics.
    Assumes target (y) is in LOG scale and converts to EUROS for reporting.
    """
    
    # 1. Predictions
    pred_train_log = model.predict(X_train)
    pred_val_log = model.predict(X_val)
    
    # 2. Revert to Real Scale (Euros)
    y_train_real = np.expm1(y_train)
    y_val_real = np.expm1(y_val)
    pred_train_real = np.expm1(pred_train_log)
    pred_val_real = np.expm1(pred_val_log)
    
    # 3. Calculate Metrics
    r2_train = r2_score(y_train, pred_train_log)
    r2_val = r2_score(y_val, pred_val_log)
    
    rmse_train = np.sqrt(mean_squared_error(y_train_real, pred_train_real))
    rmse_val = np.sqrt(mean_squared_error(y_val_real, pred_val_real))
    
    # 4. Calculate the Gap
    rmse_gap_perc = ((rmse_val - rmse_train) / rmse_train) * 100
    r2_drop = r2_train - r2_val
    
    # 5. Display Table
    df_metrics = pd.DataFrame({
        "Metric": ["RMSE (Error in Euro)", "R2 (Score)"],
        "Train": [f"{rmse_train:.2f}Euro", f"{r2_train:.4f}"],
        "Validation": [f"{rmse_val:.2f}Euro", f"{r2_val:.4f}"],
        "Difference (Gap)": [f"+{rmse_gap_perc:.1f}%", f"-{r2_drop:.4f}"]
    })
    
    print(f"\nOVERFIT ANALYSIS: {model_name.upper()}")
    print("="*60)
    print(df_metrics.to_string(index=False))
    print("-" * 60)
    
    # 6. Automatic Verdict
    if r2_val > r2_train:
        print("EXCELLENT: The model generalizes better than in training (slight underfitting or luck).")
    elif rmse_gap_perc < 10 and r2_drop < 0.03:
        print("HEALTHY: The model is robust. Validation error is close to training.")
    elif rmse_gap_perc < 20:
        print("WARNING: Some Overfitting. The model is starting to memorize training data.")
    else:
        print("DANGER: Severe Overfitting! The model memorized training and fails in validation.")
        
    return df_metrics


def predict_group(df, model, scaler, mapping, g_mean, train_cols):
    if df.empty: return pd.DataFrame()
    
    df_enc = df.copy()
    # Target Encode
    df_enc["Brand_model_encoded"] = df_enc.apply(lambda x: mapping.get((x["Brand"], x["model"]), g_mean), axis=1)
    
    # One Hot
    df_enc = pd.get_dummies(df_enc, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    
    # Alignment
    X_test = df_enc.reindex(columns=train_cols, fill_value=0)
    
    # Scaling
    X_test_s = scaler.transform(X_test)
    
    # Predict and invert log
    preds = np.expm1(model.predict(X_test_s))
    
    return pd.DataFrame({"carID": df["carID"], "price": preds})


def train_and_evaluate_rf(train_df, val_df, group_name):
    # A. Target Encoding
    mapping = train_df.groupby(["Brand", "model"])["price_log"].mean().to_dict()
    global_mean = train_df["price_log"].mean()
    
    # B. Encoding & Prep
    train_len = len(train_df)
    combined = pd.concat([train_df, val_df], axis=0)
    combined["Brand_model_encoded"] = combined.apply(
        lambda x: mapping.get((x["Brand"], x["model"]), global_mean), axis=1
    )
    combined = pd.get_dummies(combined, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    
    drop_cols = ["price", "price_log", "carID", "model", "previousOwners"]
    X_train = combined.iloc[:train_len].drop(columns=drop_cols, errors='ignore')
    y_train = combined.iloc[:train_len]["price_log"]
    X_val = combined.iloc[train_len:].drop(columns=drop_cols, errors='ignore')
    y_val = combined.iloc[train_len:]["price_log"]
    
    train_cols = X_train.columns
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=train_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=train_cols)
    
    # C. Randomized Search
    X_comb = pd.concat([X_train_s, X_val_s], axis=0).reset_index(drop=True)
    y_comb = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    ps = PredefinedSplit([-1]*len(X_train_s) + [0]*len(X_val_s))
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [500, 1000],
        'max_depth': [10, 15, 20, 30, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4)
    }
    
    search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=50, 
        cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42
    )
    
    print(f"\nTraining Group: {group_name}")
    search.fit(X_comb, y_comb)
    best_rf_split = search.best_estimator_
    
    # D. Overfit Analysis
    p_train = best_rf_split.predict(X_train_s)
    p_val = best_rf_split.predict(X_val_s)
    
    r2_train = r2_score(y_train, p_train)
    r2_val = r2_score(y_val, p_val)
    
    rmse_train = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(p_train)))
    rmse_val = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(p_val)))
    
    print(f"[{group_name}] R2 Train: {r2_train:.4f} | R2 Validation: {r2_val:.4f}")
    print(f"[{group_name}] RMSE Train: {rmse_train:.2f}Euro | RMSE Validation: {rmse_val:.2f}Euro")
    
    gap = (rmse_val - rmse_train) / rmse_train * 100
    print(f"[{group_name}] Overfit Gap: {gap:.2f}%")
    
    return best_rf_split, scaler, mapping, global_mean, train_cols


def train_and_evaluate_et(train_df, val_df, group_name):
    mapping = train_df.groupby(["Brand", "model"])["price_log"].mean().to_dict()
    global_mean = train_df["price_log"].mean()
    
    train_len = len(train_df)
    combined = pd.concat([train_df, val_df], axis=0)
    
    combined["Brand_model_encoded"] = combined.apply(
        lambda x: mapping.get((x["Brand"], x["model"]), global_mean), axis=1
    )
    
    combined = pd.get_dummies(combined, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    
    drop_cols = ["price", "price_log", "carID", "model", "previousOwners"]
    X_train = combined.iloc[:train_len].drop(columns=drop_cols, errors='ignore')
    y_train = combined.iloc[:train_len]["price_log"]
    X_val = combined.iloc[train_len:].drop(columns=drop_cols, errors='ignore')
    y_val = combined.iloc[train_len:]["price_log"]
    
    train_cols = X_train.columns
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=train_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=train_cols)
    
    X_comb = pd.concat([X_train_s, X_val_s], axis=0).reset_index(drop=True)
    y_comb = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    ps = PredefinedSplit([-1]*len(X_train_s) + [0]*len(X_val_s))
    
    et_model = ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=False)
    
    param_dist = {
        'n_estimators': [500],
        'max_features': [1.0],
        'max_depth': [30],
        'min_samples_leaf': [1],
        'min_samples_split': [10]
    }
    
    search = RandomizedSearchCV(
        estimator=et_model, 
        param_distributions=param_dist, 
        n_iter=50, 
        cv=ps, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    print(f"\nTraining ExtraTrees - Group: {group_name}")
    search.fit(X_comb, y_comb)
    best_et_split = search.best_estimator_
    print(f"Best Params ({group_name}): {search.best_params_}")
    
    p_train = best_et_split.predict(X_train_s)
    p_val = best_et_split.predict(X_val_s)
    
    r2_train = r2_score(y_train, p_train)
    r2_val = r2_score(y_val, p_val)
    
    rmse_train = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(p_train)))
    rmse_val = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(p_val)))
    
    print(f"[{group_name}] R2 Train: {r2_train:.4f} | R2 Validation: {r2_val:.4f}")
    print(f"[{group_name}] RMSE Train: {rmse_train:.2f}Euro | RMSE Validation: {rmse_val:.2f}Euro")
    
    gap = (rmse_val - rmse_train) / rmse_train * 100
    print(f"[{group_name}] Overfit Gap: {gap:.2f}%")
    
    return best_et_split, scaler, mapping, global_mean, train_cols


def train_and_evaluate_hgb(train_df, val_df, group_name):
    mapping = train_df.groupby(["Brand", "model"])["price_log"].mean().to_dict()
    global_mean = train_df["price_log"].mean()
    
    train_len = len(train_df)
    combined = pd.concat([train_df, val_df], axis=0)
    
    combined["Brand_model_encoded"] = combined.apply(
        lambda x: mapping.get((x["Brand"], x["model"]), global_mean), axis=1
    )
    
    combined = pd.get_dummies(combined, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    
    drop_cols = ["price", "price_log", "carID", "model", "previousOwners"]
    X_train = combined.iloc[:train_len].drop(columns=drop_cols, errors='ignore')
    y_train = combined.iloc[:train_len]["price_log"]
    X_val = combined.iloc[train_len:].drop(columns=drop_cols, errors='ignore')
    y_val = combined.iloc[train_len:]["price_log"]
    
    train_cols = X_train.columns
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=train_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=train_cols)
    
    X_comb = pd.concat([X_train_s, X_val_s], axis=0).reset_index(drop=True)
    y_comb = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    ps = PredefinedSplit([-1]*len(X_train_s) + [0]*len(X_val_s))
    
    hgb_model = HistGradientBoostingRegressor(
        loss='squared_error',
        random_state=42,
        early_stopping=False 
    )
    
    param_dist = {
        'l2_regularization': [0.5], 
        'max_leaf_nodes': [80], 
        'max_depth': [10], 
        'learning_rate': [0.05],
        'max_iter': [3000],
        'max_features': [0.3] 
    }
    
    search = RandomizedSearchCV(
        estimator=hgb_model, 
        param_distributions=param_dist, 
        n_iter=50, 
        cv=ps, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    print(f"\nTraining HistGB - Group: {group_name}")
    search.fit(X_comb, y_comb)
    best_hgb_split = search.best_estimator_
    print(f"Best Params ({group_name}): {search.best_params_}")
    
    p_train = best_hgb_split.predict(X_train_s)
    p_val = best_hgb_split.predict(X_val_s)
    
    r2_train = r2_score(y_train, p_train)
    r2_val = r2_score(y_val, p_val)
    
    rmse_train = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(p_train)))
    rmse_val = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(p_val)))
    
    print(f"[{group_name}] R2 Train: {r2_train:.4f} | R2 Validation: {r2_val:.4f}")
    print(f"[{group_name}] RMSE Train: {rmse_train:.2f}Euro | RMSE Validation: {rmse_val:.2f}Euro")
    
    gap = (rmse_val - rmse_train) / rmse_train * 100
    print(f"[{group_name}] Overfit Gap: {gap:.2f}%")
    
    return best_hgb_split, scaler, mapping, global_mean, train_cols


def train_stacking(train_df, val_df, test_df, estimators, group_name="Group"):
    """
    Generic Stacking function.
    estimators: List of tuples with models [(name, model), ...]
    """
    print(f"\nSTARTING STACKING (5-Fold CV): {group_name}")
    
    full_train = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    
    mapping = full_train.groupby(["Brand", "model"])["price_log"].mean().to_dict()
    global_mean = full_train["price_log"].mean()
    
    def apply_encoding_and_prep(df):
        d = df.copy()
        d["Brand_model_encoded"] = d.apply(lambda x: mapping.get((x["Brand"], x["model"]), global_mean), axis=1)
        d = pd.get_dummies(d, columns=["Brand", "transmission", "fuelType"], drop_first=True)
        return d

    df_full_proc = apply_encoding_and_prep(full_train)
    df_test_proc = apply_encoding_and_prep(test_df)
    
    drop_cols = ["price", "price_log", "carID", "model", "previousOwners"]
    train_cols = df_full_proc.drop(columns=drop_cols, errors='ignore').columns
    
    X = df_full_proc[train_cols]
    y = df_full_proc["price_log"]
    X_test = df_test_proc.reindex(columns=train_cols, fill_value=0)
    
    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)
    X_test_s = scaler.transform(X_test)
    
    stack_model = StackingRegressor(
        estimators=estimators, 
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1,
        passthrough=False,
        verbose=1
    )
    
    print(f"   Training Stack (Involves training {len(estimators)} models x 5 folds)...")
    stack_model.fit(X_s, y)
    
    print(f"   Stacking Completed!")
    try:
        coefs = stack_model.final_estimator_.coef_
        names = [name for name, _ in estimators]
        print(f"   Meta-Model Weights: {list(zip(names, coefs))}")
    except:
        pass

    preds_log = stack_model.predict(X_test_s)
    preds_final = np.expm1(preds_log)
    
    return pd.DataFrame({
        "carID": test_df["carID"].values,
        "price": preds_final
    })


def prepare_data_for_ensemble(train_df, val_df):
    mapping = train_df.groupby(["Brand", "model"])["price_log"].mean().to_dict()
    global_mean = train_df["price_log"].mean()
    
    combined = pd.concat([train_df, val_df], axis=0)
    combined["Brand_model_encoded"] = combined.apply(
        lambda x: mapping.get((x["Brand"], x["model"]), global_mean), axis=1
    )
    combined = pd.get_dummies(combined, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    
    drop_cols = ["price", "price_log", "carID", "model", "previousOwners"]
    train_len = len(train_df)
    
    X = combined.drop(columns=drop_cols, errors='ignore')
    y = combined["price_log"]
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train_s = X_scaled.iloc[:train_len]
    X_val_s = X_scaled.iloc[train_len:]
    y_train = y.iloc[:train_len]
    y_val = y.iloc[train_len:]
    
    X_final = pd.concat([X_train_s, X_val_s], axis=0).reset_index(drop=True)
    y_final = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    
    split_index = ([-1] * len(X_train_s)) + ([0] * len(X_val_s))
    ps = PredefinedSplit(test_fold=split_index)
    
    return X_final, y_final, ps, scaler, mapping, global_mean, X_train_s.columns


def predict_ensemble(df, model, scaler, mapping, g_mean, train_cols):
    if df.empty: return pd.DataFrame()
    df_enc = df.copy()
    df_enc["Brand_model_encoded"] = df_enc.apply(lambda x: mapping.get((x["Brand"], x["model"]), g_mean), axis=1)
    df_enc = pd.get_dummies(df_enc, columns=["Brand", "transmission", "fuelType"], drop_first=True)
    X_test = df_enc.reindex(columns=train_cols, fill_value=0)
    X_test_s = scaler.transform(X_test)
    preds = np.expm1(model.predict(X_test_s))
    return pd.DataFrame({"carID": df["carID"], "price": preds})


def optimize_ensemble_weights(models_dict, X, y, ps, group_name):
    estimators_list = list(models_dict.items())
    model_names = [name for name, _ in estimators_list]
    n_models = len(estimators_list)
    
    random.seed(42)
    weights_combinations = []
    for _ in range(100): 
        w = []
        for _ in range(n_models):
            if random.random() < 0.2: w.append(0) 
            else: w.append(random.randint(1, 100))
        if sum(w) == 0: w[0] = 50
        weights_combinations.append(w)
    weights_combinations.append([1] * n_models) 
    
    voting_base = VotingRegressor(estimators=estimators_list, n_jobs=-1)
    param_grid = {'weights': weights_combinations}
    
    search = RandomizedSearchCV(
        estimator=voting_base,
        param_distributions=param_grid,
        n_iter=50,
        cv=ps,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    print(f"\n Optimizing Weights for {group_name}...")
    search.fit(X, y)
    
    best_weights = search.best_params_['weights']
    print(f" Best RMSE ({group_name}): {-search.best_score_:.5f}")
    print(f" Ideal Weights: {list(zip(model_names, best_weights))}")
    
    return search.best_estimator_


def clean_data(df):
    df = df.copy()
    
    df = df.drop(columns=["hasDamage", "paintQuality%"], errors='ignore')
    
    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].apply(lambda x: x.str.lower() if x.dtype=="object" else x)
    for col in df.select_dtypes(include="object").columns:
        df = f.fix_typos(col, df)

    df['transmission'] = df['transmission'].replace('other', 'unknown')
    df['fuelType'] = df['fuelType'].replace('other', 'electric')

    df.loc[df["mileage"] < 0, "mileage"] = np.nan
    df.loc[~df["tax"].between(0, 600), "tax"] = np.nan
    df.loc[~df["mpg"].between(0, 150), "mpg"] = np.nan
    df.loc[~df["engineSize"].between(1, 6.3), "engineSize"] = np.nan
    df.loc[~df["year"].between(1990, 2020), "year"] = np.nan
    df.loc[~df["previousOwners"].between(0, 6), "previousOwners"] = np.nan
    
    mask = (df['year'] % 1 != 0)
    df.loc[mask, 'year'] = np.nan
    mask = (df['mileage'] % 1 != 0)
    df.loc[mask, 'mileage'] = np.nan
    mask = (df['tax'] % 1 != 0)
    df.loc[mask, 'tax'] = np.nan
    mask = df['mpg'] != df['mpg'].round(1)
    df.loc[mask, 'mpg'] = np.nan
    mask = df['engineSize'] != df['engineSize'].round(1)
    df.loc[mask, 'engineSize'] = np.nan

    df['mileage'] = np.log1p(df['mileage'])
    df['mpg'] = np.log1p(df['mpg'])
    df['tax'] = np.log1p(df['tax'])
    
    df["previousOwners"] = pd.to_numeric(df["previousOwners"], errors='coerce').round().astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors='coerce').round().astype("Int64")
    
    df = f.fill_NaN_with_categorical(df, "Brand", ["model", "transmission", "fuelType"])
    df = f.fill_NaN_with_categorical(df, "Brand", ["model", "transmission"])
    df = f.fill_NaN_with_categorical(df, "model", ["Brand", "transmission", "fuelType"])
    df = f.fill_NaN_with_categorical(df, "model", ["Brand", "transmission"])
    df = f.fill_NaN_with_categorical(df, "mpg", ["model", "fuelType"])
    
    df["transmission"] = df["transmission"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    df["fuelType"] = df["fuelType"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

    df = f.fill_NaN_with_mixed(df, "year", "model", "mileage")
    df = f.fill_NaN_with_mixed(df, "mileage", "model", "year")
    df = f.fill_NaN_with_mixed(df, "tax", "model", "year")
    df = f.fill_NaN_with_mixed(df, "engineSize", "model", "tax")

    df["previousOwners"] = df["previousOwners"].transform(lambda x: x.fillna(x.median())).round().astype("Int64")
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(["carID", "price"], errors='ignore')
    for col in numeric_cols:
        df[col] = df[col].astype(float)
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        if "Int64" in str(df[col].dtype):
            df[col] = df[col].round().astype("Int64")
            
    return df