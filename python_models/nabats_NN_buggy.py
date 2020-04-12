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


def is_buggy(n):
    if n >= 1:
        return 1.0
    return 0.0


def build_dataset(name):

    df = pd.read_csv("../files/nabats_dataset.csv")

    #Shuffle the row ordering
    df = df.sample(frac=1).reset_index(drop=True)

    #Make a binary buggy column
    df['buggy'] = df['num_bugs'].apply(lambda b : is_buggy(b))

    print(df[['buggy','num_bugs']].head(300))

    #Specify input ande output columns
    X = df[['CC','LOC','churn']]
    y = df['num_bugs']

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



nn_name = "nabats_buggy"


if os.path.exists("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl"):
    X_train,X_test,y_train,y_test = load_pickles(nn_name)

else:
    X_train,X_test,y_train,y_test = build_dataset(nn_name)


nepochs = 0
input_dimensions = 3
output_dimensions = 1

model = Sequential()
model.add(Dense(45, input_dim=input_dimensions,activation='relu'))
model.add(Dense(45, activation='relu'))
model.add(Dense(output_dimensions,activation='sigmoid'))
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
