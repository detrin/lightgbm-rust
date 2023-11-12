use lightgbm::{Booster, Dataset};
use serde_json::json;
use eval_metrics::error::EvalError;
use eval_metrics::classification::RocCurve;
use std::time::Instant;
use clap::{arg, Command};
use polars::prelude::*; 
use rayon::prelude::*;  
use csv::StringRecord;  

#[warn(dead_code)]
fn load_file_simple(file_path: &str) -> (Vec<Vec<f64>>, Vec<f32>) {  
    let mut rdr = csv::ReaderBuilder::new()  
        .has_headers(true)  
        .delimiter(b',')  
        .from_path(file_path)  
        .unwrap();  
  
    // Read all records into a Vec  
    let records: Vec<StringRecord> = rdr.records().map(|r| r.unwrap()).collect();  
      
    // Process records in parallel  
    let results: Vec<(Vec<f64>, f32)> = records.par_iter().map(|record| {  
        let label = record[record.len()-1].parse::<f32>().unwrap();  
        let feature: Vec<f64> = record  
            .iter()  
            .take(record.len() - 1)  
            .map(|x| x.parse::<f64>().unwrap())  
            .collect();  
        (feature, label)  
    }).collect();  
  
    // Split results into separate vectors  
    let features: Vec<Vec<f64>> = results.par_iter().map(|(f, _)| f.clone()).collect();  
    let labels: Vec<f32> = results.par_iter().map(|(_, l)| *l).collect();  
  
    (features, labels)  
}     

fn transpose<T: Default + Clone>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {  
    let mut res = vec![vec![T::default(); v.len()]; v[0].len()];  
    for (i, vi) in v.iter().enumerate() {  
        for (j, vij) in vi.iter().enumerate() {  
            res[j][i] = vij.clone();  
        }  
    }  
    res  
}  

fn load_file(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f32>), Box<dyn std::error::Error>> {  
    // Read the CSV file into a DataFrame  
    let df = CsvReader::from_path(file_path)?  
        .has_header(true)  
        .finish()?;  
  
    // Convert the DataFrame into a 2D Vec<f64> for features and a Vec<f32> for labels  
    let features: Vec<Vec<f64>> = df  
        .drop("y")?  
        .iter()  
        .map(|series| {  
            match series.f64() {
                Ok(s) => s.into_no_null_iter().collect::<Vec<f64>>(),
                Err(error) => panic!("Problem with parsing a column: {:?}", error),
            }
        })  
        .collect::<Vec<Vec<f64>>>();
    let features = transpose(features);  
  
    let labels: Vec<f32> = df  
        .column("y")?  
        .cast(&DataType::Float32)?
        .f32()?.into_no_null_iter().collect::<Vec<f32>>();
  
    Ok((features, labels))  
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
    let (train_features, train_labels) = load_file_simple(train_input);
    let (test_features, test_labels) = load_file_simple(test_input);
    // let (train_features, train_labels) = match load_file(train_input) {
    //     Ok((train_features, train_labels)) => (train_features, train_labels),
    //     Err(error) => panic!("Problem with loading training data: {:?}", error),
    // };
    // let (test_features, test_labels) = match load_file(test_input) {
    //     Ok((test_features, test_labels)) => (test_features, test_labels),
    //     Err(error) => panic!("Problem with loading test data: {:?}", error),
    // };
    let duration_load = start.elapsed();

    let params = json! {
        {
            "num_iterations": 100,
            "objective": "binary",
            "metric": "auc"
        }
    };

    let start = Instant::now();
    let train_dataset = Dataset::from_mat(train_features.clone(), train_labels.clone()).unwrap();
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