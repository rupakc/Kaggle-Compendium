import pandas as pd
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def get_multioutput_classifier(classifier):
    multi = MultiOutputClassifier(estimator=classifier)
    return [multi], ["Multioutput Classifier"]


def get_multiclass_classifier(base_estimator):
    output_code = OutputCodeClassifier(base_estimator,random_state=42)
    one_vs_one = OneVsOneClassifier(base_estimator)
    one_vs_all = OneVsRestClassifier(base_estimator)
    return [output_code,one_vs_one,one_vs_all], ['Output Code','One Vs One','One Vs All']


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


def spilt_date(list_of_date_string,separator='-',format='yyyy-mm-dd'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    for date_string in list_of_date_string:
        date_list = str(date_string).strip().split(separator)
        month_list.append(date_list[1])
        day_list.append(date_list[2])
        year_list.append(date_list[0])
    return month_list,day_list,year_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name, ' ---------------\n'
    predicted_values = trained_model.predict(X_test)
    print predicted_values[0]
    print '---------'
    print y_test[0]
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


reco_frame = pd.read_csv('train_ver2.csv')
reco_value_frame = reco_frame[list(reco_frame.columns[25:49])]
columns_reco = list(range(21,44))
impute = Imputer()
reco_frame.drop(reco_frame.columns[[5,8,11,15]],axis=1,inplace=True)
reco_frame.drop(reco_frame.columns[columns_reco],axis=1,inplace=True)
del reco_frame['fecha_dato']
del reco_frame['fecha_alta']

encoded_frame = label_encode_frame(reco_frame)
encoded_frame = encoded_frame.head(1000)
imputed_values = impute.fit_transform(encoded_frame.values)
rf = RandomForestClassifier(n_estimators=101,min_samples_split=5,min_samples_leaf=7,random_state=42)
X_train,X_test,y_train,y_test = train_test_split(imputed_values,reco_value_frame.values,test_size=0.2,random_state=42)
classifier_list, classifier_name_list = get_multiclass_classifier(rf)
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
