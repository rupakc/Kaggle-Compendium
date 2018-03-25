import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=51,max_depth=5,min_samples_split=3,random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    bag = BaggingClassifier(n_estimators=51,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=51,random_state=42,max_depth=5)
    classifier_list = [rf,grad,ada,bag,extra]
    classifier_name_list = ['Random Forests','Gradient Boosting','AdaBoost','Bagging','Extra Trees']
    return classifier_list,classifier_name_list


def evaluate_models(trained_model,trained_model_name,X_test,y_test):
    predicted_values = trained_model.predict(X_test)
    print ('---------- For Model Name : ', trained_model_name, ' ---------\n')
    print (metrics.classification_report(y_test,predicted_values))
    print (metrics.accuracy_score(y_test,predicted_values))
    print (metrics.matthews_corrcoef(y_test,predicted_values))
    print ('-------------------------------------\n')


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


train_frame = pd.read_csv('train.csv', sep=';')
del train_frame['Nr. rooms']
encoded_frame = label_encode_frame(train_frame)
target_class_labels = list(encoded_frame['Score'].values)
del train_frame['Score']
feature_values = encoded_frame.values
X_train,X_test,y_train,y_test = train_test_split(feature_values,target_class_labels,test_size=0.1,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    evaluate_models(classifier,classifier_name,X_test,y_test)
