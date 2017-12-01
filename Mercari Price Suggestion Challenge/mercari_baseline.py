import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Conv1D,MaxPooling1D,LSTM,Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_gaussian_process_regressor():
    gp = GaussianProcessRegressor()
    return [gp],['Gaussian Process']


def get_mlp_regressor(num_hidden_units=51):
    mlp = MLPRegressor(hidden_layer_sizes=num_hidden_units)
    return [mlp],['Multi-Layer Perceptron']


def get_ensemble_models():
    rf = RandomForestRegressor(n_estimators=51,min_samples_leaf=5,min_samples_split=3,random_state=42)
    bag = BaggingRegressor(n_estimators=51,random_state=42)
    extra = ExtraTreesRegressor(n_estimators=71,random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    grad = GradientBoostingRegressor(n_estimators=101,random_state=42)
    classifier_list = [rf,bag,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list, classifier_name_list


def get_linear_model():
    elastic_net = ElasticNet()
    return [elastic_net],['Elastic Net']


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name ,' ---------\n'
    predicted_values = trained_model.predict(X_test)
    print "Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Absolute Error : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score : ", metrics.r2_score(y_test,predicted_values)
    print "---------------------------------------\n"


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is np.nan:
            for i in range(len(dataframe)):
                if i > 1000:
                    break
                if type(dataframe[column][i]) is str:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def get_one_hot_category(list_of_values,n=100):
    encoded_list = list([])
    for value in list_of_values:
        encoded_list.append(one_hot(value,n))
    return pad_sequences(np.array(encoded_list),maxlen=n)


def get_deep_nn(input_feature_dim):
    model = Sequential()
    model.add(Dense(1000,input_dim=input_feature_dim,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


def get_rnn_model(timesteps,feature_dim):
    model = Sequential()
    model.add(LSTM(100,input_shape=(timesteps,feature_dim),dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
    model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


def get_cnn_model(input_feature_dim):
    model = Sequential()
    model.add(Conv1D(32,kernel_size=3,input_shape=(input_feature_dim,1),padding='same',activation='relu'))
    model.add(Conv1D(16,kernel_size=3,padding='same'))
    model.add(MaxPooling1D(pool_size=2,padding='same'))
    model.add(Conv1D(32,kernel_size=3,padding='same',activation='relu'))
    model.add(Conv1D(16,kernel_size=3,padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


filename = 'train.tsv'
train_frame = pd.read_csv(filename,sep='\t')
train_frame.dropna(inplace=True)
columns_to_encode = ['name','category_name','brand_name','item_description']
target_values = np.array(train_frame['price'].values)
del train_frame['train_id']
del train_frame['price']

category_name_encoding = get_one_hot_category(list(train_frame['category_name'].values))
name_encoding = get_one_hot_category(list(train_frame['name'].values))
brand_name_encoding = get_one_hot_category(list(train_frame['brand_name'].values))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_frame['item_description'].values))
sequences = tokenizer.texts_to_sequences(list(train_frame['item_description'].values))
item_description_encoding = pad_sequences(sequences,maxlen=200)

train_frame.drop(columns_to_encode,axis=1,inplace=True)

X = np.hstack((category_name_encoding,name_encoding,brand_name_encoding,item_description_encoding,train_frame.values))
X = X.reshape(X.shape[0],1,X.shape[1])
y = target_values.reshape(target_values.shape[0],1)

model = get_rnn_model(1,X.shape[2])
model.fit(X,y,epochs=25,batch_size=500,validation_split=0.33)


