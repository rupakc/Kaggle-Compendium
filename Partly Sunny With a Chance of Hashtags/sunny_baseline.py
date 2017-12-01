import pandas as pd
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv1D,LSTM,GRU,Bidirectional,MaxPooling1D,Embedding,Flatten
from keras.preprocessing.sequence import pad_sequences


def get_deep_nn(input_dim,output_dim,vocab_dimension):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_dimension,output_dim=125,input_length=input_dim))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(75,activation='relu',activity_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


def get_deep_cnn(input_dim,output_dim,vocab_dimension):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_dimension, output_dim=125, input_length=input_dim))
    model.add(Conv1D(32,3,activation='relu',padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(16,3,activation='relu',padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32,3,activation='relu',padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(output_dim,activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def get_deep_rnn(input_dim,output_dim,vocab_dimension):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_dimension, output_dim=125, input_length=input_dim))
    model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(50),merge_mode='ave'))
    model.add(Dropout(0.2))
    model.add(Dense(55,activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(output_dim,activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


train_frame = pd.read_csv('train.csv')
tweet_list = list(train_frame['tweet'].values)
train_frame.drop(['id','tweet','state','location'],axis=1,inplace=True)
dimension = 2000
one_hot_encoded_tweet_list = list([])
for tweet in tweet_list:
    one_hot_tweet = one_hot(tweet,dimension)
    one_hot_encoded_tweet_list.append(one_hot_tweet)

padded_tweet_list = pad_sequences(one_hot_encoded_tweet_list)

X = padded_tweet_list
y = train_frame.values
model = get_deep_rnn(X.shape[1],y.shape[1],dimension)
print model.summary()
model.fit(X,y,epochs=2,batch_size=1000,validation_split=0.33,verbose=2)

