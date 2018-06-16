import hashfeatures
import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


def get_classifiers():
    rf = RandomForestClassifier(n_estimators=51, random_state=42)
    linear_svm = LinearSVC()
    classifier_list = [rf,linear_svm]
    classifier_name_list = ['Random Forest','Linear SVM']
    return classifier_list,classifier_name_list


def report_classification_metrics(trained_model, X_test, y_test):
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print metrics.accuracy_score(y_test,predicted_values)


script_file_path = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Seinfield Scripts\\scripts.csv'
script_frame = pd.read_csv(script_file_path)
character_group_series = script_frame['Character'].value_counts()
filtered_character_list = []
for character, count in character_group_series.iteritems():
    if count > 300:
        filtered_character_list.append(character)

filtered_script_frame = script_frame[script_frame['Character'].isin(filtered_character_list)]
del script_frame
character_list = list(filtered_script_frame['Character'].values)
dialogue_list = preprocess.text_clean_pipeline_list(list(filtered_script_frame['Dialogue'].values))
hash_feature_set = hashfeatures.FeatureHash(max_feature_num=1000).get_feature_set(dialogue_list)
del filtered_script_frame
X_train,X_test,y_train,y_test = train_test_split(hash_feature_set,character_list,test_size=0.2,random_state=42)
del character_list
del dialogue_list
del hash_feature_set
classifier_list,classifier_name_list = get_classifiers()
for classifier,classifier_name in zip(classifier_list, classifier_name_list):
    classifier.fit(X_train,y_train)
    report_classification_metrics(classifier,X_test,y_test)
