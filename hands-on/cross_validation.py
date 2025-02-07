import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="\n%(levelname)s: %(message)s")

# ----------------------------------------------------------

training_data = pd.read_csv("./data_sources/train.csv")
logger.info("Training data loaded successfully")

X = training_data.drop("SalePrice", axis=1)
y = training_data.SalePrice

# Drop non numerical columns
X_drop = X.select_dtypes(exclude=["object"])

my_pipeline = Pipeline(
    steps=[
        ("preprocessor", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=50, random_state=0)),
    ]
)
logger.info("Created pipeline for preproccessing and model")

scores = -1 * cross_val_score(
    my_pipeline, X_drop, y, cv=5, scoring="neg_mean_absolute_error"
)
logger.info(f"Got scores {scores}")
logger.info(f"Avg {(sum(scores) / len(scores))}")
