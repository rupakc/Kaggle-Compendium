import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


def get_regressor_list():
    rf = RandomForestRegressor(n_estimators=51, random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    grad = GradientBoostingRegressor(random_state=42)
    bag = BaggingRegressor(n_estimators=42)
    return [rf, ada, grad, bag], ['Random Forest', 'Adaboost', 'Gradientboost', 'Bagging']


def parse_dates(date_list,date_sep='-'):
    year_list = list([])
    month_list = list([])
    day_list = list([])
    for date_string in date_list:
        date_array = date_string.split(date_sep)
        year_list.append(int(date_array[0]))
        month_list.append(int(date_array[1]))
        day_list.append(int(date_array[2]))
    return year_list, month_list, day_list


filepath = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Store Item Forecasting\\train.csv'
train_frame = pd.read_csv(filepath, error_bad_lines=False)
sales_value = list(train_frame['sales'].values)
train_frame['year'], train_frame['month'], train_frame['day'] = parse_dates(train_frame['date'].values)
del train_frame['date']
del train_frame['sales']

X_train, X_test, y_train, y_test = train_test_split(train_frame.values, sales_value, test_size=0.2, random_state=42)

del train_frame
del sales_value

classifier_list, classifier_name_list = get_regressor_list()
for classifier, classifier_name in zip(classifier_list, classifier_name_list):
    print '----- For Classifier : ', classifier_name, ' --------------\n'
    classifier.fit(X_train, y_train)
    predicted_values = classifier.predict(X_test)
    print metrics.r2_score(y_test, predicted_values)
    print metrics.mean_squared_error(y_test, predicted_values)
    print metrics.median_absolute_error(y_test, predicted_values)
    print '-----------------------\n'
