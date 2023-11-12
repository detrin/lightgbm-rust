import polars as pl
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import json

datasets = []
for n_features in [10, 100, 1000]:
    for n_samples in [10**3, 10**4, 10**5, 10**6]:
        if n_samples == 1000000 and n_features == 1000:
            continue
        X, y = make_blobs(
            n_samples=n_samples * 2,
            centers=2,
            n_features=n_features,
            random_state=1,
            cluster_std=10.0,
            center_box=(-3.0, 3.0),
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=1
        )
        data = pl.DataFrame({"x{}".format(i): X_train[:, i] for i in range(n_features)})
        data = data.with_columns(pl.Series("y", y_train).cast(pl.Float32))
        data.write_csv(f"./data/data_f{n_features}_s{n_samples}_train.csv")
        # data.write_parquet(f"./data/data_f{n_features}_s{n_samples}.parquet")
        data = pl.DataFrame({"x{}".format(i): X_test[:, i] for i in range(n_features)})
        data = data.with_columns(pl.Series("y", y_test).cast(pl.Float32))
        data.write_csv(f"./data/data_f{n_features}_s{n_samples}_test.csv")
        datasets.append(
            {
                "n_features": n_features,
                "n_samples": n_samples,
                "train_input": f"./data/data_f{n_features}_s{n_samples}_train.csv",
                "test_input": f"./data/data_f{n_features}_s{n_samples}_test.csv",
            }
        )

with open("./data/datasets.json", "w") as f:
    json.dump(datasets, f)
