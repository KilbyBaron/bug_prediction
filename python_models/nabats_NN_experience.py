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



def build_dataset(name):

    df = pd.read_csv("../files/nabats_dataset.csv")

    #Drop non-buggy rows
    df = df.loc[df['num_bugs']>0]

    #Shuffle the row ordering
    df = df.sample(frac=1).reset_index(drop=True)

    #Get the average exp per bug in each file
    df['exp'] = df['exp']/df['num_bugs']

    #Specify input ande output columns
    X = df[['CC','LOC','churn']]
    y = df['exp']

    #Split into train and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

    #Save train and test sets
    X_train.to_pickle("../files/nn_training/pickle_barrel/X_train_"+name+".pkl")
    X_test.to_pickle("../files/nn_training/pickle_barrel/X_test_"+name+".pkl")
    y_train.to_pickle("../files/nn_training/pickle_barrel/y_train_"+name+".pkl")
    y_test.to_pickle("../files/nn_training/pickle_barrel/y_test_"+name+".pkl")

    return X_train,X_test,y_train,y_test

def load_pickles(name):
    X_train = pd.read_pickle("../files/nn_training/pickle_barrel/X_train_"+name+".pkl")
    X_test = pd.read_pickle("../files/nn_training/pickle_barrel/X_test_"+name+".pkl")
    y_train = pd.read_pickle("../files/nn_training/pickle_barrel/y_train_"+name+".pkl")
    y_test = pd.read_pickle("../files/nn_training/pickle_barrel/y_test_"+name+".pkl")

    return X_train,X_test,y_train,y_test



nn_name = "nabats_experience"


if os.path.exists("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl"):
    X_train,X_test,y_train,y_test = load_pickles(nn_name)

else:
    X_train,X_test,y_train,y_test = build_dataset(nn_name)


nepochs = 1000
input_dimensions = 3
output_dimensions = 1
hidden_layer_size = 100

model = Sequential()
model.add(Dense(hidden_layer_size, input_dim=input_dimensions,activation='relu'))
model.add(Dense(hidden_layer_size, activation='relu'))
model.add(Dense(hidden_layer_size, activation='relu'))
model.add(Dense(output_dimensions,activation='linear'))
model.compile(loss="mean_squared_error", optimizer='adam')


model.load_weights("../files/nn_training/models/NN_"+nn_name+".h5")


#Train model
csv_logger = CSVLogger("../files/nn_training/training/nn_training_"+nn_name+".csv", append=True)

model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=nepochs, verbose=1, callbacks=[csv_logger])
model.save("../files/nn_training/models/NN_"+nn_name+".h5")


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
