import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="\n%(levelname)s: %(message)s")


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# ----------------------------------------------------------
# Part 1. Load data

train_data = pd.read_csv("./data_sources/train.csv")
logger.info("Data loaded successfully")

y = train_data.SalePrice
X = train_data.drop(["SalePrice"], axis=1)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

X_train_original = X_train_full.copy()
X_valid_original = X_valid_full.copy()

cols_with_missing = [
    col for col in X_train_full.columns if X_train_full[col].isnull().any()
]

logger.warning("Found columns with missing values:\n%s", cols_with_missing)

X_train_full.drop(cols_with_missing, axis=1)
X_valid_full.drop(cols_with_missing, axis=1)

logger.info("Dropped columns with missing values")

low_cardinality_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "object"
]

logger.info("Low cardinality columns:\n%s", low_cardinality_cols)

numerical_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].dtype in ["int64", "float64"]
]

logger.info("Numerical columns:\n%s", numerical_cols)

# Keep selected columns only
picked_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[picked_cols].copy()
X_valid = X_valid_full[picked_cols].copy()

# Get list of categorical variables
s = X_train.dtypes == "object"
categorical_cols = list(s[s].index)

logger.info("Categorical variables:\n%s", categorical_cols)

# ----------------------------------------------------------
# Part 2. Check Approach 1 - Drop columns with categorical columns
logger.info("Approach 1 - Drop cols with categorical data")

drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

appr_1_score = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
logger.info("MAE: %s", appr_1_score)

# ----------------------------------------------------------
# Part 3. Check Approach 2 - Ordinal encoding
logger.info("Approach 2 - Ordinal encoding")

X_train_ordinal = X_train.copy()
X_valid_ordinal = X_valid.copy()

ordinal_encoder = OrdinalEncoder()

oe_encoded_X_train_cols = ordinal_encoder.fit_transform(X_train[categorical_cols])
X_train_ordinal[categorical_cols] = oe_encoded_X_train_cols

oe_encoded_X_valid_cols = ordinal_encoder.fit_transform(X_valid[categorical_cols])
X_valid_ordinal[categorical_cols] = oe_encoded_X_valid_cols

appr_2_score = score_dataset(X_train_ordinal, X_valid_ordinal, y_train, y_valid)
logger.info("MAE: %s", appr_2_score)

# ----------------------------------------------------------
# Part 4. Check Approach 3 - One hot encoding
logger.info("Approach 3 - One Hot encoding")

OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(categorical_cols, axis=1)
num_X_valid = X_valid.drop(categorical_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

logger.info("MAE: %s", score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
