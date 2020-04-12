import pandas as pd
import numpy as np
import math
import os



def short_path(path):
    length = 3
    path = path.split('/')
    new_path = ""

    for x in range(1,length+1):
        new_path.append




#Change to loop later
projects = ['accumulo']#, 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'hive', 'openjpa', 'pig', 'wicket']
p=projects[0]


df = pd.read_csv("../files/"+p+"/index_to_filename.csv")
metric_df = pd.read_csv("../files/all_metrics.csv")
metric_df = metric_df.loc[metric_df['project']==p]


#trim paths of each df for matching
short_path_size = 5
df['short_path'] = df['file_path'].apply( lambda path : "/".join(path.split("/")[-1*short_path_size:]))
metric_df['short_path'] = metric_df['filepath'].apply( lambda path : "/".join(path.split("/")[-1*short_path_size:]))


df['cc']=0
df['loc']=0

s0,s1,s2,s3,s4 = 0,0,0,0,0

for i,r in df.iterrows():

    #print(round((i/27213)*100),end="\r")

    for length in range(-1,-8,-1):
        sp = "/".join(r['file_path'].split("/")[length:])

        metric_row = metric_df.loc[metric_df['filepath'].str.contains(sp)]
        n = metric_row.shape[0]
        if n <= 1:
            break
        if length < -6 and n > 1:
            for i2,r2 in metric_row.iterrows():
                print(r2['cc'])
            print("\n")

    if n == 0:
        s0 += 1
    if n == 1:
        s1 += 1
    if n == 2:
        s2 += 1
    if n == 3:
        s3 += 1
    if n == 4:
        s4 += 1

print(s0,s1,s2,s3,s4)


#
