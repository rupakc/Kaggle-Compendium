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
    print "R2 Score : ", trained_model.score(X_test,y_test)
    print "---------------------------------------\n"


folder_name = 'Train_Skies/'
file_name_prefix = 'Training_Sky'
file_extension = '.csv'
final_frame = pd.read_csv(folder_name + file_name_prefix + str(1) + file_extension)
halo_frame = pd.read_csv('Training_halos.csv')
final_frame['GalaxyID'] = map(lambda x:x.replace('Galaxy',''),final_frame['GalaxyID'].values)
halo_frame['SkyId'] = map(lambda x:x.replace('Sky',''),halo_frame['SkyId'].values)

for i in range(2,301):
    filename_to_read = folder_name + file_name_prefix + str(i) + file_extension
    df = pd.read_csv(filename_to_read)
    final_frame = pd.concat([final_frame,df])

master_frame = pd.merge(final_frame,halo_frame,how='outer',left_on='GalaxyID',right_on='SkyId')
output_values = master_frame[['halo_x1','halo_y1']].values
master_frame.drop(['halo_x1','halo_y1'],axis=1,inplace=True)
del master_frame['GalaxyID']
del master_frame['SkyId']
master_frame_values = Imputer().fit_transform(master_frame.values)
output_values = Imputer().fit_transform(output_values)
X_train,X_test,y_train,y_test = train_test_split(master_frame_values,output_values,test_size=0.2,random_state=42)
regressor_list, regressor_name_list = get_ensemble_models()
for regressor, regressor_name in zip(regressor_list,regressor_name_list):
    multi_reg = get_multioutput_regressor(regressor)
    multi_reg.fit(X_train,y_train)
    print_evaluation_metrics(multi_reg,regressor_name,X_test,y_test)
