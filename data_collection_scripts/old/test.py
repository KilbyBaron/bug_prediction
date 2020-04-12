import os
import re
import sys
import pandas as pd

projects = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'hive', 'openjpa', 'pig', 'wicket']

for p in projects:

    try:
        os.mkdir("../../files/clean_files/"+p)
    except:
        pass


    df = pd.read_csv("../../files/"+p+"/train_data.csv")

    f = open("../../files/clean_files/"+p+"/target_file_indexes",'w')

    indexes = set(df['index2'].to_list())

    print(len(indexes))

    for i in indexes:

        f.write(str(i)+"\n")

    f.close()
