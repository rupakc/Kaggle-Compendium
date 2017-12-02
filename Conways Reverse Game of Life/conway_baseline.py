import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten

filename = 'train.csv'
train_frame = pd.read_csv(filename)
start_column_names = list([])
stop_column_names = list([])
board_x,board_y = 20,20

for column_name in train_frame.columns:
    if column_name.find('start') != -1:
        start_column_names.append(column_name)
    if column_name.find('stop') != -1:
        stop_column_names.append(column_name)

start_features = train_frame[start_column_names].values
stop_features = train_frame[stop_column_names].values
del train_frame
stop_features = stop_features.reshape(stop_features.shape[0],board_x,board_y)
stop_features = stop_features.reshape(stop_features.shape[0],1,board_x,board_y)

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(1,board_x,board_y),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(32,(2,2),padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(16,(2,2),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(board_x* board_y,activation='softmax'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
print model.summary()

model.fit(stop_features,start_features,batch_size=1000,epochs=25,validation_split=0.2,verbose=2)

