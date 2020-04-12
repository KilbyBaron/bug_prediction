#!/usr/bin/env bash

project=$1

echo "Identifying modified methods..."
./modified_functions.sh $project

echo "Collecting target file indexes..."
python3 collect_target_file_indexes.py $project

echo "Downloading buggy files..."
./collect_buggy_methods.sh $project

echo "Converting methods to vectors with code2vec..."
./collect_c2v_vectors.sh $project

echo "Assembling training data dataframe..."
python3 make_c2v_training_data.py $project
