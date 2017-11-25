import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split


def get_rnn_model(timesteps,feature_dim):
    model = Sequential()
    model.add(LSTM(31,input_shape=(timesteps,feature_dim),dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


filename = 'train_seq.csv'
train_frame = pd.read_csv(filename)
X = train_frame['A'].values
y = train_frame['B'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
X_train = X_train.reshape(X_train.shape[0],1,1)
X_test = X_test.reshape(X_test.shape[0],1,1)

model = get_rnn_model(X_train.shape[1],X_train.shape[2])
print model.summary()
model.fit(X_train,y_train,epochs=250,batch_size=25)
print model.evaluate(X_test,y_test)
