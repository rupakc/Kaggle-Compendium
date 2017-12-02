import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np


filename = 'train.csv'
train_frame = pd.read_csv(filename)
master_sequence_list = list([])
max_float = 1000000.0
for sequence in list(train_frame['Sequence'].values):
    sequence = sequence.strip()
    sequence_list = sequence.split(',')
    sequence_list = map(lambda x:float(x)%max_float,sequence_list)
    master_sequence_list.append(sequence_list)

padded_sequence_list = pad_sequences(master_sequence_list)
X,y = padded_sequence_list[:,:-1],padded_sequence_list[:,-1]
X = np.reshape(X,(X.shape[0],X.shape[1],1))
y = np.reshape(y,(y.shape[0],1))

model = Sequential()
model.add(LSTM(100,input_shape=(X.shape[1],X.shape[2]),return_sequences=True,dropout=0.2,recurrent_dropout=0.2))
model.add(GRU(200,recurrent_dropout=0.2,dropout=0.2))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
model.fit(X,y,validation_split=0.3,epochs=25,batch_size=10000)
