

#!/usr/bin/env bash

project=$1
mkdir -p "../files/$project/buggy_files/"

file_path="../files/$project/target_file_indexes"
target_indexes=`cat $file_path`

cd "../../cloned_repos/$project"

for index in $target_indexes;
do
	#Pass file contents to python script first, this will rename all target methods to avoid confusion from overloading
	git show $index | python3 "../../c2v_models/scripts/rename_overloaded_methods.py" $index $project > "../../c2v_models/files/$project/buggy_files/$index.java"
done
