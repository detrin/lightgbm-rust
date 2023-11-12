import polars as pl
from lightgbm import LGBMClassifier  
from sklearn.metrics import roc_auc_score  
import time  
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_input", type=str, default="train.csv")
parser.add_argument("--test_input", type=str, default="test.csv")
args = parser.parse_args()

  
start = time.time()  
data = pl.read_csv(args.train_input).with_columns(
    pl.col("y").cast(pl.Int32)
)
train_features = data.select(pl.exclude("y")).to_numpy()
train_labels = data.select(pl.col("y")).to_numpy()
data = pl.read_csv(args.test_input).with_columns(
    pl.col("y").cast(pl.Int32)
)
test_features = data.select(pl.exclude("y")).to_numpy()
test_labels = data.select(pl.col("y")).to_numpy()
print(train_labels)
duration_load = time.time() - start  
  
params = {  
    "n_estimators": 100,  
    "objective": "binary",  
    "metric": "auc"  
}  
  
start = time.time()  
model = LGBMClassifier(**params)  
model.fit(train_features, train_labels)  
duration_train = time.time() - start  
  
start = time.time()  
train_preds = model.predict(train_features)  
test_preds = model.predict(test_features)  
duration_inference = time.time() - start  
  
print("Train AUC:", roc_auc_score(train_labels, train_preds))  
print("Test AUC:", roc_auc_score(test_labels, test_preds))  
results = {
    "load": duration_load,
    "train": duration_train,
    "inference": duration_inference
}

print(results)
