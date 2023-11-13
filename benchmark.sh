#!/bin/bash  
set -e  
  
results=()  
datasets=$(jq -c '.[]' ./data/datasets.json)  
total_datasets=$(jq -c '.[]' ./data/datasets.json | wc -l)  
counter=1  
for row in ${datasets[@]}; do  
    n_features=$(echo ${row} | jq -r '.n_features') 
    n_samples=$(echo ${row} | jq -r '.n_samples') 
    train_input=$(echo ${row} | jq -r '.train_input')  
    test_input=$(echo ${row} | jq -r '.test_input')  
  
    for i in 1 2 3 4 5 6 7 8 9 10; do 
        python_output=$(python3.11 lightgbm_benchmark.py --train_input ${train_input} --test_input ${test_input} | tail -n1)  
        rust_output=$(./target/release/lightgbm_rust --train_input ${train_input} --test_input ${test_input} | tail -n1)   
        results+=("{\"n_features\":${n_features},\"n_samples\":${n_samples},\"python\":${python_output},\"rust\":${rust_output},\"train_input\":\"${train_input}\",\"test_input\":\"${test_input}\"}")  
    done  
    echo "Processed ${counter} out of ${total_datasets} datasets."  

    ((counter++))  
done  
  
echo "["$(IFS=','; echo "${results[*]}")"]" > ./data/results.json  
