import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import time
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--train_input", type=str, default="train.csv")
parser.add_argument("--test_input", type=str, default="test.csv")
args = parser.parse_args()


start = time.time()
data = pl.read_csv(args.train_input)
train_features = data.select(pl.exclude("y"))
train_labels = data["y"]
data = pl.read_csv(args.test_input)
test_features = data.select(pl.exclude("y"))
test_labels = data["y"]
duration_load = time.time() - start

params = {"n_estimators": 100, "objective": "binary", "metric": "auc"}

start = time.time()
model = LGBMClassifier(**params)
model.fit(train_features, train_labels)
duration_train = time.time() - start

start = time.time()
train_preds = model.predict(train_features)
duration_inference = time.time() - start
test_preds = model.predict(test_features)

print("Train AUC:", roc_auc_score(train_labels, train_preds))
print("Test AUC:", roc_auc_score(test_labels, test_preds))
results = {
    "load": int(duration_load*1000),
    "train": int(duration_train*1000),
    "inference": int(duration_inference*1000),
}

print(json.dumps(results))
