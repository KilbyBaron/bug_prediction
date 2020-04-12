from keras.models import Sequential
from keras.layers import Dense
import keras.metrics
import pandas as pd
import numpy as np
import math
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
import os



def build_dataset():

    #Combine all projects
    projects = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'hive', 'openjpa', 'pig', 'wicket']
    dfs = []
    for p in projects:
        p_df = pd.read_csv("../files/"+p+"/train_data.csv")
        dfs.append(p_df)

    df = pd.concat(dfs)


    #Convert vector data to lists
    df['vector'] = df['vector'].apply(lambda v : v.replace('\n','').split(' '))
    df['vector'] = df['vector'].apply(lambda v : [float(i) for i in v])


    #Convert fixed (2) to non-buggy (0)
    df['buggy'] = df['buggy'].apply(lambda b : b%2)


    #Shuffle the row ordering
    df = df.sample(frac=1).reset_index(drop=True)

    #Specify input and output columns
    X = pd.DataFrame(df['vector'].to_list())
    y = df['buggy']

    #Split into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

    #Save train and test sets
    X_train.to_pickle("../files/nn_training/pickle_barrel/X_train_buggy.pkl")
    X_test.to_pickle("../files/nn_training/pickle_barrel/X_test_buggy.pkl")
    y_train.to_pickle("../files/nn_training/pickle_barrel/y_train_buggy.pkl")
    y_test.to_pickle("../files/nn_training/pickle_barrel/y_test_buggy.pkl")

    return X_train,X_test,y_train,y_test

def load_pickles(type):
    X_train = pd.read_pickle("../files/nn_training/pickle_barrel/X_train_"+type+".pkl")
    X_test = pd.read_pickle("../files/nn_training/pickle_barrel/X_test_"+type+".pkl")
    y_train = pd.read_pickle("../files/nn_training/pickle_barrel/y_train_"+type+".pkl")
    y_test = pd.read_pickle("../files/nn_training/pickle_barrel/y_test_"+type+".pkl")

    return X_train,X_test,y_train,y_test


if os.path.exists("../files/nn_training/pickle_barrel/X_train_buggy.pkl"):
    print(1)
    X_train,X_test,y_train,y_test = load_pickles('buggy')

else:
    print(2)
    X_train,X_test,y_train,y_test = build_dataset()


nepochs = 0
input_dimensions = 384
output_dimensions = 1

print(y_train.shape)
bs = 0
ns = 0
for i in y_train.iteritems():
    if i[1] == 1:
        bs += 1
    else:
        ns += 1
print("Number of buggy vs non-buggy:",bs,ns)


model = Sequential()
model.add(Dense(output_dimensions, input_dim=input_dimensions,activation='sigmoid'))
model.compile(loss="mean_squared_error", optimizer='adam')


model.load_weights("../files/nn_training/models/NN_semantic_buggy.h5")



#Train model
csv_logger = CSVLogger("../files/nn_training/training/nn_training_buggy.csv", append=True)

model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=nepochs, verbose=1, callbacks=[csv_logger])
model.save("../files/nn_training/models/NN_semantic_buggy.h5")


y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
test = list()

false_pos = 0.0
false_neg = 0.0
true_pos = 0.0
true_neg = 0.0

for i in range(len(y_pred)):

    prediction = int((y_pred[i][0]).round())
    actual = y_test.iloc[i]

    pred.append(prediction)
    test.append(actual)

    if prediction == 1:
        if actual == 1:
            true_pos += 1
        else:
            false_pos += 1
    else:
        if actual == 0:
            true_neg += 1
        else:
            false_neg += 1


print("False positives:",false_pos)
print("False negatives:",false_neg)
print("True positives:",true_pos)
print("True negatives:",true_neg)
print("precision:",true_pos/(true_pos+false_pos))
print("recall:",true_pos/(false_neg+true_pos))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
