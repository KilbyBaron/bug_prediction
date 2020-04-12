import os
import re
import sys
import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join



# This method renames a function name the same way c2v does
def c2v_renaming(name):
    new_name = ""
    for c in name:
        if c.isupper() or c == '_':
            new_name = new_name+"|"
        if c != '_':
            new_name = new_name + c.lower()
    return new_name


def extract_metrics_from_line(line):

    index, method, vector, buggy, fixed, hash, fix_size, priority, experience = 0,0,0,0,0,0,0,0,0

    m_num = 'a'

    # Determine if the method was buggy or fixed
    if "buggy" in line[1]:
        print(line[0],line[1])
        buggy = 1
        m_num = line[1].split("|buggy|")[-1][0]
    if "|fixed" in line[1]:
        fixed = 1

    # Get a few metrics straight from the line
    index = line[0].split('.')[0]
    method = line[1].replace("|buggy|"+m_num,'').replace("|fixed",'')
    vector = line[2]

    # If the method is buggy, extract the last few metrics
    if buggy == 1:

        # Find the row in function data with the matching function name, file index, and method number
        fd_row = function_data.loc[(function_data['function_name']==method)&(function_data['index1']==index)&(function_data['method_number']==m_num)]
        fd_row.reset_index(inplace=True)

        # Get hash and fix size from function_data.csv
        hash = fd_row.at[0,'hash'].replace("^!","")
        fix_size = fd_row.at[0,'fix_size']

        bfc_row = bfcs.loc[bfcs['BFC_id']==hash]
        bfc_row.reset_index(inplace=True)

        experience  = bfc_row.at[0,'author_exp']
        priority = bfc_row.at[0,'priority']

    return index, method, vector, buggy, fixed, hash, fix_size, priority, experience






#Get project name as argument
project = sys.argv[1]
path = '/home/kilby/Documents/code/c2v_models/files/'+project

#Prepare dataframe to hold training data
vector_df = pd.DataFrame(columns=['commit_index','method','vector','buggy'])


#Use function_data.csv to get the hash and fix_size of buggy modified_methods
#edit the method names in the same way code2vec does
function_data = pd.read_csv(path+'/function_data.csv')
function_data['function_name'] = function_data['function_name'].apply(lambda m : c2v_renaming(m))


#Use bfc data to get priority and experience levels
bfcs = pd.read_csv(path+'/bfcs.csv')

# As vector_df grows it takes longer to add new rows, so collect intermediate
# dfs in list and combine them at the end
dfs = []
EXPORT_DF_SIZE = 10000000000



#Loop through vector_data_output and add to DataFrame
#Export dataframe occasionally to keep speed up and data safe in case of crash
with open("../files/"+project+"/vector_data.txt",'r') as vector_file:

    #Get file length
    file_length = 0.0
    for line in vector_file:
        file_length += 1.0
    vector_file.seek(0)

    #Loop through c2v output
    line_num=0
    for line in vector_file:
        try:

            #Output progress
            line_num += 1
            progress = str(round(100*(line_num/file_length),2))+'%'+"\t Vector number:"+str(line_num)
            #print(progress,end='\r')

            #Split c2v output
            line = line.split(',')

            if len(line) ==3:

                index, method, vector, buggy, fixed, hash, fix_size, priority, experience = extract_metrics_from_line(line)

                #Add each method to the new dataframe
                vector_df = vector_df.append({
                    'commit_index':index,
                    'method':line[1],
                    'vector':line[2],
                    'buggy':buggy,
                    'fixed':fixed,
                    'hash': hash,
                    'fix_size':fix_size,
                    'priority':priority,
                    'experience':experience

                    },ignore_index=True)

            #Periodically add vector_df to a list and start fresh to save time
            if line_num % EXPORT_DF_SIZE == 0:
                dfs.append(vector_df)
                vector_df = pd.DataFrame(columns=['commit_index','method','vector','buggy'])

        except Exception as e:
            print(e)
            
dfs.append(vector_df)
train_data = pd.concat(dfs)
train_data.to_csv("../files/"+project+"/train_data_test.csv",index=False)
