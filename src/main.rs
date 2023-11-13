use lightgbm::{Booster, Dataset};
use serde_json::json;
use eval_metrics::error::EvalError;
use eval_metrics::classification::RocCurve;
use std::time::Instant;
use clap::{arg, Command};
use polars::prelude::*; 
use faer::Mat;  
use faer::polars::polars_to_faer_f64;  
  
fn load_file_faer(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f32>), Box<dyn std::error::Error>> {       
    let df = CsvReader::from_path(file_path)?    
        .has_header(true)    
        .finish()?;    
      
    let mat: Mat<f64> = polars_to_faer_f64(df.lazy()).unwrap();    
    
    let mut features: Vec<Vec<f64>> = vec![vec![0.0; mat.ncols()-1]; mat.nrows()];  
    for i in 0..mat.ncols()-1 {  
        for j in 0..mat.nrows() {  
            features[j][i] = mat[(j, i)];  
        }  
    }  
   
    let mut labels: Vec<f32> = vec![0.0; mat.nrows()];
    let y_ind = mat.ncols()-1;
    for i in 0..mat.nrows() {  
        labels[i] = mat[(i, y_ind)] as f32; 
    }   
  
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
    let (train_features, train_labels) = match load_file_faer(train_input) {
        Ok((train_features, train_labels)) => (train_features, train_labels),
        Err(error) => panic!("Problem with loading training data: {:?}", error),
    };
    let (test_features, test_labels) = match load_file_faer(test_input) {
        Ok((test_features, test_labels)) => (test_features, test_labels),
        Err(error) => panic!("Problem with loading test data: {:?}", error),
    };
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