# Goal: Download the data set from HuggingFace and save it to the data directory

# Usage:
# python scripts/acquire_data.py

# Dependencies
# ------------

# standard libraries
from pathlib import Path

# third party libraries
import pandas as pd
from datasets import load_dataset

# Paths
# -----

# construct the paths
PATH_REPO = Path(__file__).parent.parent
PATH_DATA = PATH_REPO / "data" / "raw"
PATH_TRAIN = PATH_DATA / "train"
PATH_TEST = PATH_DATA / "test"

# make directories
PATH_TRAIN.mkdir(parents=True, exist_ok=True)
PATH_TEST.mkdir(parents=True, exist_ok=True)

# Download and save the data set
# ------------------------------

print("Downloading the data set from Hugging Face...")

# load the data set
dataset = load_dataset("imodels/diabetes-readmission")

# access the data in pandas
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

# get features and labels for train and test
X_train = df_train.drop(columns=["readmitted"])
y_train = df_train["readmitted"]

X_test = df_test.drop(columns=["readmitted"])
y_test = df_test["readmitted"]

print("Saving the data as csv files to the data directory...")

# save the data as csv files to the data directory
X_train.to_csv(PATH_TRAIN / "X_train.csv", index=False)
y_train.to_csv(PATH_TRAIN / "y_train.csv", index=False)
X_test.to_csv(PATH_TEST / "X_test.csv", index=False)
y_test.to_csv(PATH_TEST / "y_test.csv", index=False)

print("Done!")
