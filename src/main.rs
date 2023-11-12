extern crate csv;
extern crate itertools;
extern crate lightgbm;
extern crate serde_json;

use polars::prelude::*;
use itertools::zip;
use lightgbm::{Booster, Dataset};
use serde_json::json;
use eval_metrics::error::EvalError;
use eval_metrics::classification::{RocCurve, RocPoint, PrCurve, PrPoint};
use std::time::{Duration, Instant};

use clap::{arg, Command};

fn main_old() {
    let df = CsvReader::from_path("./data/data_f100_s10000.csv").unwrap().finish().unwrap();

    let mut file = std::fs::File::open("./data/data_f100_s10000.parquet").unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();
}

fn load_file(file_path: &str) -> (Vec<Vec<f64>>, Vec<f32>) {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(file_path);
    let mut labels: Vec<f32> = Vec::new();
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut header = true;
    for result in rdr.unwrap().records() {
        if header {
            header = false;
            continue;
        }
        let record = result.unwrap();
        let r_len = record.len();
        let label = record[r_len-1].parse::<f32>().unwrap();
        let feature: Vec<f64> = record
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<f64>>()[0..r_len-1]
            .to_vec();
        labels.push(label);
        features.push(feature);
    }
    (features, labels)
}

fn main() -> Result<(), EvalError> {
    let matches = Command::new("lightgbm_rust")  
        .version("0.1.0")  
        .author("Daniel Herman daniel.herman@protonmail.com>")  
        .about("Testing out rust binding for LightGBM.")  
        .arg(arg!(--train_input <FILE> "Train table (CSV format)")  
            .required(true))
        .arg(arg!(--test_input <FILE> "Test table (CSV format)")  
            .required(true))
        .get_matches();

    let train_input = matches.get_one::<String>("train_input").expect("required");
    let test_input = matches.get_one::<String>("test_input").expect("required");

    let start = Instant::now();
    let (train_features, train_labels) =
        load_file(train_input);
    let (test_features, test_labels) =
        load_file(test_input);
    let train_dataset = Dataset::from_mat(train_features.clone(), train_labels.clone()).unwrap();
    let duration_load = start.elapsed();

    let params = json! {
        {
            "num_iterations": 100,
            "objective": "binary",
            "metric": "auc"
        }
    };

    let start = Instant::now();
    let booster = Booster::train(train_dataset, &params).unwrap();
    let duration_train = start.elapsed();
    let start = Instant::now();
    let train_result = booster.predict(train_features).unwrap();
    let duration_inference = start.elapsed();
    let test_result = booster.predict(test_features).unwrap();
    // println!("feature importance");
    // let feature_name = booster.feature_name().unwrap();
    // let feature_importance = booster.feature_importance().unwrap();
    // for (feature, importance) in zip(&feature_name, &feature_importance) {
    //     println!("{}: {}", feature, importance);
    // }
    // AUC
    let train_labels: Vec<bool> = train_labels.iter().map(|x| *x > 0.5).collect();
    let roc = RocCurve::compute(&train_result[0], &train_labels)?;
    println!("Train AUC: {}", roc.auc());
    let test_labels: Vec<bool> = test_labels.iter().map(|x| *x > 0.5).collect();
    let roc = RocCurve::compute(&test_result[0], &test_labels)?;
    println!("Test AUC: {}", roc.auc());

    let elapsed_time = json! {
        {
            "load": duration_load.as_millis(),
            "train": duration_train.as_millis(),
            "inference": duration_inference.as_millis()
        }
    };
    println!("{}", elapsed_time);
    Ok(())
}