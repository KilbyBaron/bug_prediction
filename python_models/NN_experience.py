from keras.models import Sequential
from keras.layers import Dense
import keras.metrics
import pandas as pd
import numpy as np
import math
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
import os
import statistics
from keras.models import load_model


def build_dataset():

    #Combine all projects
    projects = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'hive', 'openjpa', 'pig', 'wicket']
    dfs = []
    for p in projects:
        p_df = pd.read_csv("../files/"+p+"/train_data.csv")
        dfs.append(p_df)

    df = pd.concat(dfs)

    #Only use buggy methods for this training
    df = df.loc[(df['buggy']==1)&(pd.notna(df['experience']))]

    df['experience'] = df['experience'].apply(lambda exp : int(exp))

    #Convert vector data to lists
    df['vector'] = df['vector'].apply(lambda v : v.replace('\n','').split(' '))
    df['vector'] = df['vector'].apply(lambda v : [float(i) for i in v])

    #Shuffle the row ordering
    df = df.sample(frac=1).reset_index(drop=True)

    #Specify input ande output columns
    X = pd.DataFrame(df['vector'].to_list())
    y = df['experience']

    #Split into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

    #Save train and test sets
    X_train.to_pickle("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl")
    X_test.to_pickle("../files/nn_training/pickle_barrel/X_test_"+nn_name+".pkl")
    y_train.to_pickle("../files/nn_training/pickle_barrel/y_train_"+nn_name+".pkl")
    y_test.to_pickle("../files/nn_training/pickle_barrel/y_test_"+nn_name+".pkl")

    return X_train,X_test,y_train,y_test

def get_dataset():

    #If the dataset has never been made, make it
    if not os.path.exists("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl"):
        return build_dataset()

    #If it has been made, load the saved dataset
    X_train = pd.read_pickle("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl")
    X_test = pd.read_pickle("../files/nn_training/pickle_barrel/X_test_"+nn_name+".pkl")
    y_train = pd.read_pickle("../files/nn_training/pickle_barrel/y_train_"+nn_name+".pkl")
    y_test = pd.read_pickle("../files/nn_training/pickle_barrel/y_test_"+nn_name+".pkl")

    return X_train,X_test,y_train,y_test


def load_nn():
    if os.path.exists("../files/nn_training/models/NN_semantic_"+nn_name+".h5"):
        return load_model("../files/nn_training/models/NN_semantic_"+nn_name+".h5")

    input_dimensions = 384
    output_dimensions = 1
    model = Sequential()
    model.add(Dense(128, input_dim=input_dimensions,activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dimensions,activation='linear'))
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

def train(nepochs):
    #Train model
    csv_logger = CSVLogger("../files/nn_training/training/nn_training_"+nn_name+".csv", append=True)
    model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=nepochs, verbose=1, callbacks=[csv_logger])
    model.save("../files/nn_training/models/NN_semantic_"+nn_name+".h5")


def test():
    model = load_model("../files/nn_training/models/NN_semantic_"+nn_name+".h5")
    y_pred = model.predict(X_test)
    #Converting predictions to label
    pred = list()
    test = list()
    MAE = []
    MSE = []
    for i in range(len(y_pred)):
        MAE.append(abs((y_pred[i][0]) - (y_test.iloc[i])))
        MSE.append(((y_pred[i][0]) - (y_test.iloc[i]))**2)
    MAE = statistics.median(MAE)
    MSE = statistics.mean(MSE)

    print("Mean Absolute Error:",MAE)
    print("Mean Squared Error:",MSE)



nn_name = "experience"
model = load_nn()
X_train,X_test,y_train,y_test = get_dataset()
train(0)
test()
