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

    #Convert priority data for softmax output
    priority_conversion_dict = {
    1: [1,0,0,0,0],
    2: [0,1,0,0,0],
    3: [0,0,1,0,0],
    4: [0,0,0,1,0],
    5: [0,0,0,0,1]
    }

    #Get average priority for each file
    df['priority'] = (df['priority']/df['num_bugs']).apply(lambda p : int(round(p)))

    #Convert priorities to one-hot vectors
    df['priority'] = df['priority'].apply( lambda p : priority_conversion_dict[p])

    #Specify input ande output columns
    X = df[['CC','LOC','churn']]
    y = pd.DataFrame(df['priority'].to_list())

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



nn_name = "nabats_priority"


if os.path.exists("../files/nn_training/pickle_barrel/X_train_"+nn_name+".pkl"):
    X_train,X_test,y_train,y_test = load_pickles(nn_name)

else:
    X_train,X_test,y_train,y_test = build_dataset(nn_name)


nepochs = 1000
input_dimensions = 3
output_dimensions = 5
hidden_layer_size = 100

model = Sequential()
model.add(Dense(hidden_layer_size, input_dim=input_dimensions,activation='relu'))
model.add(Dense(hidden_layer_size, activation='relu'))
model.add(Dense(hidden_layer_size, activation='relu'))
model.add(Dense(output_dimensions,activation='softmax'))
model.compile(loss="mean_squared_error", optimizer='adam')


model.load_weights("../files/nn_training/models/NN_"+nn_name+".h5")


#Train model
csv_logger = CSVLogger("../files/nn_training/training/nn_training_"+nn_name+".csv", append=True)

model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=nepochs, verbose=1, callbacks=[csv_logger])
model.save("../files/nn_training/models/NN_"+nn_name+".h5")


#Convert priority data for softmax output
priority_conversion_dict = {
1: [1,0,0,0,0],
2: [0,1,0,0,0],
3: [0,0,1,0,0],
4: [0,0,0,1,0],
5: [0,0,0,0,1]
}
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
test = list()
for i in range(len(y_pred)):
    #if y_test.iloc[i].to_list() != [0,0,1,0,0]:
    pred.append(priority_conversion_dict[np.argmax(y_pred[i])+1])
    test.append(y_test.iloc[i].to_list())


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
