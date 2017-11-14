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
                if i > 50000:
                    break
                if type(dataframe[column][i]) is unicode or type(dataframe[column][i]) is bool or type(dataframe[column][i]) is str:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is unicode or type(dataframe[column][0]) is bool or type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


filename = 'world_cup_data.csv'
cup_frame = pd.read_csv(filename)
columns_to_delete = ['ELO Rating','FIFA  Rating','Position']
target_values = cup_frame['FIFA  Rating'].values
cup_frame.drop(columns_to_delete,axis=1,inplace=True)
cup_frame = label_encode_frame(cup_frame)

X_train,X_test,y_train,y_test = train_test_split(cup_frame.values,target_values,test_size=0.2,random_state=42)
regressor_list,regressor_name_list = get_ensemble_models()
for regressor, regressor_name in zip(regressor_list,regressor_name_list):
    regressor.fit(X_train,y_train)
    print_evaluation_metrics(regressor,regressor_name,X_test,y_test)

