import pandas as pd
from keras.layers import LSTM,GRU,Bidirectional,Dropout,BatchNormalization,Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def parse_date(date_list,separator='.'):
    day_list = list([])
    month_list = list([])
    year_list = list([])
    for date_string in date_list:
        date_array = date_string.split(separator)
        day_list.append(date_array[0])
        month_list.append(date_array[1])
        year_list.append(date_array[2])
    return day_list,month_list,year_list


def get_rnn_model(timesteps,feature_dimension):
    model = Sequential()
    model.add(LSTM(128,input_shape=(timesteps,feature_dimension),return_sequences=True,recurrent_dropout=0.3,dropout=0.4))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(256,dropout=0.4,recurrent_dropout=0.2),merge_mode='ave'))
    model.add(BatchNormalization())
    model.add(Dense(1,activation='linear',activity_regularizer='l2'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


filename = 'sales_train_v2.csv'
sale_frame = pd.read_csv(filename)

sale_frame['day'],sale_frame['month'],sale_frame['year'] = parse_date(list(sale_frame['date'].values))
del sale_frame['date']
del sale_frame['date_block_num']
predicted_values = sale_frame['item_cnt_day'].values
del sale_frame['item_cnt_day']

sale_frame['shop_id'] = MinMaxScaler().fit_transform(sale_frame['shop_id'].values.reshape(-1,1))
sale_frame['item_id'] = MinMaxScaler().fit_transform(sale_frame['item_id'].values.reshape(-1,1))
sale_frame['item_price'] = MinMaxScaler().fit_transform(sale_frame['item_price'].values.reshape(-1,1))
sale_frame['year'] = MinMaxScaler().fit_transform(sale_frame['year'].values.reshape(-1,1))

sale_frame_values = sale_frame.values.reshape(sale_frame.values.shape[0],1,sale_frame.values.shape[1])
model = get_rnn_model(sale_frame_values.shape[1],sale_frame_values.shape[2])
model.fit(sale_frame_values,predicted_values,batch_size=1024,epochs=25,validation_split=0.33)
