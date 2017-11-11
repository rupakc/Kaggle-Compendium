import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import hashfeatures
import numpy as np


def get_naive_bayes_models():
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    classifier_list = [gnb,mnb,bnb]
    classifier_name_list = ['Gaussian NB','Multinomial NB','Bernoulli NB']
    return classifier_list,classifier_name_list


def get_neural_network(hidden_layer_size=50):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size)
    return [mlp], ['MultiLayer Perceptron']


def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=51,min_samples_leaf=5,min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=71,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=57,random_state=42)
    ada = AdaBoostClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=101,random_state=42)
    classifier_list = [rf,bagg,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list,classifier_name_list


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def get_full_hash_features(dataframe):
    title_features = hashfeatures.FeatureHash(max_feature_num=20).get_feature_set(list(dataframe['title'].values))
    desc_features = hashfeatures.FeatureHash(max_feature_num=450).get_feature_set(list(dataframe['description'].values))
    attr_features = hashfeatures.FeatureHash(max_feature_num=100).get_feature_set(list(dataframe['attrs'].values))
    dataframe.drop(['title','description','attrs'],axis=1,inplace=True)
    full_features = np.hstack((title_features,desc_features,attr_features,dataframe.values))
    return full_features


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


filename = 'avito_train.tsv'
avito_frame = pd.read_csv(filename,sep='\t',nrows=20000)
columns_to_delete = ['itemid','is_proved','close_hours']
target_label_name = 'is_blocked'
avito_frame['category'] = LabelEncoder().fit_transform(avito_frame['category'].values)
avito_frame['subcategory'] = LabelEncoder().fit_transform(avito_frame['subcategory'].values)
avito_frame.drop(columns_to_delete,axis=1,inplace=True)
avito_frame.dropna(inplace=True)
avito_frame['attrs'] = map(lambda x: " ".join(x.split(':')),list(avito_frame['attrs'].values))
target_class_labels = list(avito_frame[target_label_name].values)
del avito_frame[target_label_name]
features = get_full_hash_features(avito_frame)
X_train,X_test,y_train,y_test = train_test_split(features,target_class_labels,test_size=0.2,random_state=42)
smote = RandomOverSampler(random_state=42)
X_train,y_train = smote.fit_sample(X_train,y_train)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)


