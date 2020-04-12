project=$1
output_file="../../c2v_models/files/$project/index_to_filename.csv"
echo "hash,file_path,index">$output_file

cd "../../cloned_repos/$project"

awk -F "\"*,\"*" '{print}' "../../c2v_models/files/$project/train_data.csv" | while read -r ROW

do

    index=$(echo "$ROW" |cut -d ',' -f6 )
    git rev-list --objects --all | grep $index | tr ' ' , | while read line; do echo "$line,$index"; done >> $output_file

done


exit 0
