import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------
# Part 1. Load data

train_data = pd.read_csv("./data_sources/train.csv")

# select target
y = train_data.SalePrice

# select features
features = train_data.drop(["SalePrice"], axis=1)
X = features.select_dtypes(exclude=["object"])

# split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

X_train_original = X_train.copy()
X_val_original = X_val.copy()


def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


logger.info("Data loaded successfully")

# ----------------------------------------------------------
# Part 2. Check for missing values

logger.info("Checking for missing values")
cols_w_missing_vals = []
for col in X_train.columns:
    if X_train[col].isnull().any():
        cols_w_missing_vals.append(col)

logger.info("Columns with missing values: %s", cols_w_missing_vals)

# ----------------------------------------------------------
# Part 3. Check approach 1 - Drop columns with missing values

logger.info("Approach 1 - Drop columns with missing values")
dropped_X_train = X_train.drop(cols_w_missing_vals, axis=1).copy()
dropped_X_val = X_val.drop(cols_w_missing_vals, axis=1).copy()

logger.info("MAE: %s", score_dataset(dropped_X_train, dropped_X_val, y_train, y_val))

# ----------------------------------------------------------
# Part 4. Check approach 2 - Imputation

logger.info("Approach 2 - Simple imputation")
mean_imputer = SimpleImputer(strategy="mean")

imputed_X_train_mean = pd.DataFrame(mean_imputer.fit_transform(X_train))
imputed_X_val_mean = pd.DataFrame(mean_imputer.transform(X_val))

# Imputation removed column names; put them back
imputed_X_train_mean.columns = X_train.columns
imputed_X_val_mean.columns = X_val.columns

score_mean = score_dataset(imputed_X_train_mean, imputed_X_val_mean, y_train, y_val)

logger.info("MAE (mean): %s", score_mean)

median_imputer = SimpleImputer(strategy="median")

imputed_X_train_median = pd.DataFrame(median_imputer.fit_transform(X_train))
imputed_X_val_median = pd.DataFrame(median_imputer.transform(X_val))

# Imputation removed column names; put them back
imputed_X_train_median.columns = X_train.columns
imputed_X_val_median.columns = X_val.columns

score_median = score_dataset(
    imputed_X_train_median, imputed_X_val_median, y_train, y_val
)

logger.info("MAE (median): %s", score_median)

mode_imputer = SimpleImputer(strategy="most_frequent")

imputed_X_train_mode = pd.DataFrame(mode_imputer.fit_transform(X_train))
imputed_X_val_mode = pd.DataFrame(mode_imputer.transform(X_val))

# Imputation removed column names; put them back
imputed_X_train_mode.columns = X_train.columns
imputed_X_val_mode.columns = X_val.columns

score_mode = score_dataset(imputed_X_train_mode, imputed_X_val_mode, y_train, y_val)

logger.info("MAE (mode): %s", score_mode)

# ----------------------------------------------------------
# Part 5. Check approach 3 - Extended Imputation
logger.info("Approach 3 - Extended imputation")

for col in cols_w_missing_vals:
    X_train[col + "_was_missing"] = X_train[col].isnull()
    X_val[col + "_was_missing"] = X_val[col].isnull()

# ---- whats happening here ----
# Conceptually, this is what's happening:
# for col in cols_w_missing_vals:
#     # This line does something like:
#     missing_flags = []
#     for value in X_train[col]:
#         missing_flags.append(pd.isna(value))  # Check each individual value
#
#     X_train[col + "_was_missing"] = missing_flags

# Pandas `.isnull()` method automatically does this iteration for you behind the scenes. It's vectorized, meaning:
# - It goes through EVERY ROW automatically
# - Checks each value in that column
# - Creates a boolean Series in one go
# - No manual row-by-row looping needed

# Visual example:
# df = pd.DataFrame({
#     'LotArea': [5000, np.nan, 6000, np.nan, 7000]
# })
#
# # This does row-wise checking automatically
# missing_flags = df['LotArea'].isnull()
#
# print(missing_flags)
# # Output:
# # 0    False
# # 1     True
# # 2    False
# # 3     True
# # 4    False

imputed_X_train_ext = pd.DataFrame(mean_imputer.fit_transform(X_train))
imputed_X_val_ext = pd.DataFrame(mean_imputer.transform(X_val))

imputed_X_train_ext.columns = X_train.columns
imputed_X_val_ext.columns = X_val.columns

score_imp_ext = score_dataset(imputed_X_train_ext, imputed_X_val_ext, y_train, y_val)
logger.info("MAE: %s", score_imp_ext)
