import os
import re
import sys
import pandas as pd
import numpy as np


'''

This script will clean up the modified_function.csv dataframe
and export 2 files:

'''

project = sys.argv[1]

#Working directory
dir = os.getcwd()
os.chdir(dir)

# Open up output from modified_functions.sh
# and remove file creations and deletions
df = pd.read_csv("../files/"+project+"/modified_functions.csv")
df = df.loc[(df['index1'] != '000000000')&(df['index2'] != '000000000')]


#Number the methods of buggy files
df['method_number'] = 'a'
df = df.sort_values(by=['index1','function_name'])

alphabet = list(map(chr, range(97, 123)))
m_num = 0
cur_m = ""
cur_i = ""

for i,r in df.iterrows():
    if r['function_name'] == cur_m and r['index1'] == cur_i:
        m_num += 1

    else:
        m_num = 0
        cur_m = r['function_name']
        cur_i = r['index1']

    if m_num < 26:
        df.at[i,'method_number']=alphabet[m_num]


print(df.shape)

df.to_csv("../files/"+project+"/function_data.csv", index=False)

# Export list of file_indexes to collect
file_indexes = set(df['index1'].tolist() + df['index2'].tolist())
with open("../files/"+project+"/target_file_indexes",'w') as f:
  f.write('\n'.join(file_indexes))
