# Goal: Train a final model, save it, and evaluate it on the test set
# use the best parameters found in the model tuning notebook

# dependencies
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# paths
PATH_DATA = Path("../data/processed")
PATH_TRAIN = PATH_DATA / "train"
PATH_TEST = PATH_DATA / "test"
PATH_MODELS = Path("../models")

# load processed data
X_train = pd.read_csv(PATH_TRAIN / "X_train.csv")
y_train = pd.read_csv(PATH_TRAIN / "y_train.csv")
X_test = pd.read_csv(PATH_TEST / "X_test.csv")
y_test = pd.read_csv(PATH_TEST / "y_test.csv")

# set random seed
np.random.seed(42)

# initialize model
model = RandomForestClassifier(
    max_depth=7,
    min_samples_split=10,
    n_estimators=200,
    n_jobs=-1
)

# fit model
model.fit(X_train, y_train.values.ravel())

# save model
joblib.dump(model, PATH_MODELS / "final_model.joblib")

# evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC AUC score: {roc_auc_score(y_test, y_pred):.4f}")