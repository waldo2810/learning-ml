import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="\n%(levelname)s: %(message)s")


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# ----------------------------------------------------------
# Part 1. Load data

training_data = pd.read_csv("./data_sources/train.csv")
logger.info("Training data loaded successfully")

X = training_data.drop("SalePrice", axis=1)
y = training_data.SalePrice

X_train_full, X_val_full, y_train, y_val = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
logger.info("Split into training an validation data")

categorical_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "object"
]

numerical_cols = [
    col
    for col in X_train_full.columns
    if X_train_full[col].dtype in ["int64", "float64"]
]
logger.info("Identified categorical and numerical columns")

wanted_cols = categorical_cols + numerical_cols
X_train = X_train_full[wanted_cols].copy()
X_valid = X_val_full[wanted_cols].copy()
logger.info("Selected columns to work with")

# ----------------------------------------------------------
# Part 2. Create pipeline
# 2.1. Create preprocessing steps
numerical_transformer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
logger.info("Created preprocessing steps")

# 2.2. Bundle preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)
logger.info("Bundled preprocessing steps")

# Part 3. Test pipeline
# 3.1. Create model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# 3.2. Bundle preprocessing and modeling code in a pipeline
preprocess_and_model_pip = Pipeline(
    steps=[("preprocess", preprocessor), ("model", model)]
)
logger.info("Created preprocessing and modeling")

# 3.3. Preprocessing of training data, fit model
preprocess_and_model_pip.fit(X_train, y_train)
logger.info("Fit training data onto model in pipeline")

# 3.4. Preprocessing of validation data, get predictions
preds = preprocess_and_model_pip.predict(X_valid)
logger.info("Predict using pipeline")

score = mean_absolute_error(y_val, preds)
logger.info(f"MAE: {score}")
