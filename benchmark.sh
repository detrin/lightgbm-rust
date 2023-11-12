#!/bin/bash  
set -e  
  
# Initialize an empty array for results  
results=()  
  
# Read the datasets.json file  
datasets=$(jq -c '.[]' ./data/datasets.json)  
  
# Calculate total number of datasets  
total_datasets=$(jq -c '.[]' ./data/datasets.json | wc -l)  
  
# Initialize a counter for progress tracking  
counter=1  
  
# Loop over each dataset  
for row in ${datasets[@]}; do  
    n_features=$(echo ${row} | jq -r '.n_features') 
    n_samples=$(echo ${row} | jq -r '.n_samples') 
    train_input=$(echo ${row} | jq -r '.train_input')  
    test_input=$(echo ${row} | jq -r '.test_input')  
  
    for i in 1 2 3 4 5 6 7 8 9 10; do
        # Run the Python script and capture the output  
        python_output=$(python3.11 lightgbm_benchmark.py --train_input ${train_input} --test_input ${test_input} | tail -n1)  
    
        # Run the Rust binary and capture the output  
        rust_output=$(./target/release/lightgbm_rust --train_input ${train_input} --test_input ${test_input} | tail -n1)  
    
        # Append the outputs to the results array  
        results+=("{\"n_features\":${n_features},\"n_samples\":${n_samples},\"python\":${python_output},\"rust\":${rust_output},\"train_input\":\"${train_input}\",\"test_input\":\"${test_input}\"}")  
    done
    # Print progress information  
    echo "Processed ${counter} out of ${total_datasets} datasets."  
  
    # Increment the counter  
    ((counter++))  
done  
  
# Write the results to results.json  
echo "["$(IFS=','; echo "${results[*]}")"]" > ./data/results.json  
