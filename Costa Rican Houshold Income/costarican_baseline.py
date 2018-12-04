import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def get_classifier_list():
    rf = RandomForestClassifier(n_estimators=51, random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    bag = BaggingClassifier(n_estimators=42)
    return [rf, ada, grad, bag], ['Random Forest', 'Adaboost', 'Gradientboost', 'Bagging']


file_path = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Costa Rican Houshold Income\\train.csv'
train_frame = pd.read_csv(file_path, error_bad_lines=False)

del train_frame['Id']
del train_frame['idhogar']
del train_frame['dependency']
del train_frame['edjefe']
del train_frame['edjefa']
train_frame.interpolate(inplace=True)
train_frame.dropna(inplace=True)
target_values = list(train_frame['Target'].values)
del train_frame['Target']
X_train, X_test, y_train, y_test = train_test_split(train_frame.values, target_values, test_size=0.2)
classifier_list, classifier_names = get_classifier_list()
for classifier, classifier_name in zip(classifier_list, classifier_names):
    classifier.fit(X_train, y_train)
    predicted_values = classifier.predict(X_test)
    print '--------- For Classifier : ', classifier_name, ' ----------\n'
    print classification_report(y_test, predicted_values)
    print accuracy_score(y_test, predicted_values)
    print '-----------------------------------\n'
