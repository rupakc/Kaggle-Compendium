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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np


def get_multioutput_regressor(base_estimator):
    multi = MultiOutputRegressor(base_estimator)
    return multi


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
                if type(dataframe[column][i]) is unicode or type(dataframe[column][i]) is np.bool_ or type(dataframe[column][i]) is str:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is unicode or type(dataframe[column][0]) is np.bool_ or type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


filename_train = 'train.csv'
client_filename = 'cliente_tabla.csv'
product_filename = 'producto_tabla.csv'
town_filename = 'town_state.csv'

train_frame = pd.read_csv(filename_train,nrows=100000)
client_frame = pd.read_csv(client_filename)
product_frame = pd.read_csv(product_filename)
town_frame = pd.read_csv(town_filename)
target_column_name = 'Demanda_uni_equil'

train_product_frame = pd.merge(train_frame,product_frame,how='outer',left_on='Producto_ID',right_on='Producto_ID')
train_product_client_frame = pd.merge(train_product_frame,client_frame,how='outer',left_on='Cliente_ID',right_on='Cliente_ID')

unused_frames = [train_frame,client_frame,product_frame,town_frame,train_product_frame,train_product_client_frame]

master_frame = pd.merge(train_product_client_frame,town_frame,how='outer',left_on='Agencia_ID',right_on='Agencia_ID')
columns_to_delete = ['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID']
master_frame.drop(columns_to_delete,axis=1,inplace=True)

for frame in unused_frames:
    del frame

master_frame = label_encode_frame(master_frame)
master_frame.dropna(inplace=True)
target_values = list(master_frame[target_column_name].values)
del master_frame[target_column_name]

X_train,X_test,y_train,y_test = train_test_split(master_frame.values,target_values,test_size=0.2,random_state=42)
classifier_list, classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)


