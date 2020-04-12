import os
import re
import sys
import pandas as pd



if __name__ == '__main__':


    project = sys.argv[2]

    df = pd.read_csv('/home/kilby/Documents/code/c2v_models/files/'+project+'/function_data.csv')

    # Rows where the current file index is in the buggy column
    df_buggy = df.loc[df['index1']==sys.argv[1]]

    #rows where the current file index is in the fixed column
    df_fixed = df.loc[df['index2']==sys.argv[1]]

    out_string = ""
    chunk_id = ""

    #TEST
    actual_buggy = 0
    actual_fixed = 0

    for line in sys.stdin:

        # If we hit a function declaration known to be buggy, change its name
        for i,r in df_buggy.iterrows():

            #Use the chunk data to differentiate between methods with the same name
            if str(r['chunk']).replace('<comma>',',') in line:
                chunk_id = r['chunk']

            if r['function_line'].replace('<comma>',',') in line and r['chunk'] == chunk_id:
                line = line.replace(r['function_name'],r['function_name']+"_buggy_"+r['method_number'])
                actual_buggy += 1



        # If we hit a function declaration known to be fixed, change its name
        for i,r in df_fixed.iterrows():

            #Use the chunk data to differentiate between methods with the same name
            if str(r['chunk']).replace('<comma>',',') in line:
                chunk_id = r['chunk']

            if r['function_line'].replace('<comma>',',') in line and r['chunk'] == chunk_id:
                line = line.replace(r[1],r[1]+"_fixed")
                actual_fixed += 1

        out_string += line


with open("inspect.txt",'a') as f:
    f.write(str(actual_buggy)+'\n')

sys.stdout.write(out_string)
