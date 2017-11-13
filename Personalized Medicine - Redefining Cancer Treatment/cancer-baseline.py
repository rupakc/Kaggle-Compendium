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
import numpy as np
import hashfeatures
import preprocess


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
        if type(dataframe[column][0]) is np.nan:
            for i in range(len(dataframe)):
                if i > 100000:
                    break
                if type(dataframe[column][i]) is str or type(dataframe[column][i]) is np.bool_:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is str or type(dataframe[column][0]) is np.bool_:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


def get_data_frame(text_filename):
    text_list = list([])
    with open(text_filename,'r') as text_file:
        train_text_list = text_file.read().splitlines()
    for index,train_text in enumerate(train_text_list):
        if index > 0:
            index = train_text.find('||')
            if index != -1:
                text = train_text[index+2:]
                id = int(train_text[:index])
                text_list.append(dict({"ID":id,"Text":text}))
    return pd.DataFrame(text_list)


text_filename = 'training_text'
mutation_filename = 'training_variants'
text_frame = get_data_frame(text_filename)
mutation_frame = pd.read_csv(mutation_filename)
mutation_frame['ID'] = map(lambda x:int(x),mutation_frame['ID'].values)
final_frame = pd.merge(text_frame,mutation_frame,left_on='ID',right_on='ID',how='outer')
class_labels = list(final_frame['Class'].values)
gene_text = list(final_frame['Text'].values)
gene_features = hashfeatures.FeatureHash(max_feature_num=5000).get_feature_set(gene_text)
del final_frame['Class']
del final_frame['Text']
del final_frame['ID']
final_frame = label_encode_frame(final_frame)
final_feature_set = np.hstack((gene_features,final_frame.values))
X_train,X_test,y_train,y_test = train_test_split(final_feature_set,class_labels,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)

