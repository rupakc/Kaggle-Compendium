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
from keras.layers import Dense,Dropout,Conv1D,Flatten,MaxPool1D,LSTM,GRU
import numpy as np


def get_dense_mlp(input_dim):
    model = Sequential()
    model.add(Dense(51,input_dim=input_dim,activation='relu',activity_regularizer='l1'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


def get_cnn_model(input_dim):
    model = Sequential()
    model.add(Conv1D(32,3,padding='same',input_shape=(input_dim,1),activation='relu'))
    model.add(Conv1D(32,3,padding='same',activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


def get_rnn_model(timesteps,feature_dim):
    model = Sequential()
    model.add(LSTM(50,input_shape=(timesteps,feature_dim),dropout=0.2,recurrent_dropout=0.25,return_state=False))
    #model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


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


filename = 'train.csv'
train_frame = pd.read_csv(filename)
train_frame.dropna(inplace=True)
del train_frame['customer_ID']
del train_frame['time']
target_values = train_frame['cost'].values
del train_frame['cost']
train_frame = label_encode_frame(train_frame)
X_train,X_test,y_train,y_test = train_test_split(train_frame.values,target_values,test_size=0.2,random_state=42)

# model = get_dense_mlp(X_train.shape[1])
# print model.summary()
# model.fit(X_train,y_train,epochs=25,batch_size=36)
# print model.evaluate(X_test,y_test)


# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
# model = get_cnn_model(X_train.shape[1])
# print model.summary()
# model.fit(X_train,y_train,epochs=25,batch_size=36)
# print model.evaluate(X_test,y_test)
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
model = get_rnn_model(1,22)
print model.summary()
model.fit(X_train,y_train,epochs=25,batch_size=36)
print model.evaluate(X_test,y_test)

