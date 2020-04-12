#!/usr/bin/env bash

project=$1
output_file="../files/$project/modified_functions.csv"


mkdir -p "../files/$project"
echo 'index1,index2,function_name,fix_size,file_path,hash,function_line,chunk' > $output_file
python3 "make_project_bfcs_file.py" $project

cd "../../cloned_repos/$project"

awk -F "\"*,\"*" '{print}' "../../c2v_models/files/$project/bfcs.csv" | while read -r ROW

do

    proj=$(echo "$ROW" |cut -d ',' -f3 )
    proj=$(echo "$proj" | awk '{print tolower($0)}')

    hash=$(echo "$ROW" |cut -d ',' -f1 )
    hash="${hash}^!"

    git diff $hash | python3 "../../c2v_models/scripts/modified_methods.py" $hash >> "../../c2v_models/files/$project/modified_functions.csv"

    git diff --no-prefix -U10000 $hash | python3 "../../c2v_models/scripts/save_diff_methods.py" $hash

done


exit 0
