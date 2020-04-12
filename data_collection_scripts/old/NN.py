from keras.models import Sequential
from keras.layers import Dense
import keras.metrics
import pandas as pd
import numpy as np
import math
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split



'''
#Combine all projects
projects = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'felix']#, 'hive', 'openjpa', 'pig', 'wicket']
dfs = []
for p in projects:
    p_df = pd.read_csv("../files/"+p+"/train_data_test.csv")

    #shuffle the order of both
    buggy = p_df.loc[p_df['buggy']==1].sample(frac=1)
    clean = p_df.loc[p_df['buggy']==0].sample(frac=1)

    dfs.append(buggy)
    dfs.append(clean[:buggy.shape[0]])

    print(p,buggy.shape[0])


df = pd.concat(dfs)
print(df.shape)


df.to_csv("../files/NN_input_test.csv",index=False)
'''

#df = pd.read_csv("../files/NN_input_test.csv")

df = pd.read_csv("../files/accumulo/train_data_test2.csv")

#Convert vector data to lists
df['vector'] = df['vector'].apply(lambda v : v.replace('\n','').split(' '))
df['vector'] = df['vector'].apply(lambda v : [float(i) for i in v])

#Shuffle the row ordering
df = df.sample(frac=1).reset_index(drop=True)

X = pd.DataFrame(df['vector'].to_list())
y = df['buggy']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

nepochs = 2000
input_dimensions = 384
output_dimensions = 1


model = Sequential()
model.add(Dense(output_dimensions, input_dim=input_dimensions,activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(output_dimensions,activation='linear'))
model.compile(loss="mean_squared_error", optimizer='adam')
#model.load_weights("../files/NN_semantic_1.h5")



#Train model
csv_logger = CSVLogger("nn_training.csv", append=True)

model.fit(X_train,y_train,validation_data = (X_test,y_test), epochs=nepochs, verbose=1, callbacks=[csv_logger])
model.save("../files/NN_semantic_1.h5")

y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
test = list()
for i in range(len(y_pred)):
    pred.append(int(y_pred[i][0].round()))
    test.append(y_test.iloc[i])


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)
