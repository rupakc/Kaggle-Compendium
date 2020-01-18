import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import regression
from sklearn.preprocessing import LabelEncoder


def get_regressors():
    rf = RandomForestRegressor(n_estimators=51, random_state=42)
    extra = ExtraTreesRegressor(n_estimators=71, random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    grad = GradientBoostingRegressor(random_state=42)
    regressor_list = [rf, extra, ada, grad]
    regressor_name_list = ['Random Forest', 'Extra Trees', 'AdaBoost', 'GradientBoost']
    return regressor_list, regressor_name_list


def evaluate_model(predicted_values, actual_values):
    print('---------------------------------\n')
    print('R2 Score : %f' % regression.r2_score(actual_values, predicted_values))
    print('Mean Absoulte Error : %f' % regression.mean_absolute_error(actual_values, predicted_values))
    print('Median Absolute Error: %f' % regression.median_absolute_error(actual_values, predicted_values))
    print('----------------------------------\n')


def parse_date(date_list,separator='-'):
    year_list = []
    month_list = []
    day_list = []
    for date_string in date_list:
        parsed_date_list = date_string.split(separator)
        year_list.append(int(parsed_date_list[0]))
        month_list.append(int(parsed_date_list[1]))
        day_list.append(int(parsed_date_list[2]))
    return year_list, month_list, day_list


avocado_frame = pd.read_csv('avocado.csv', error_bad_lines=False)
del avocado_frame['Unnamed: 0']
avocado_frame['type'] = LabelEncoder().fit_transform(avocado_frame['type'].values.reshape(-1, 1))
avocado_frame['region'] = LabelEncoder().fit_transform(avocado_frame['region'].values.reshape(-1, 1))
target_values = list(avocado_frame['AveragePrice'].values)
del avocado_frame['AveragePrice']
avocado_frame['year'], avocado_frame['month'], avocado_frame['day'] = parse_date(list(avocado_frame['Date'].values))
del avocado_frame['Date']
X_train, X_test, y_train, y_test = train_test_split(avocado_frame.values, target_values, test_size=0.2, random_state=42)
regressor_list, regressor_name_list = get_regressors()
for regressor, regressor_name in zip(regressor_list, regressor_name_list):
    regressor.fit(X_train, y_train)
    predicted_values = regressor.predict(X_test)
    print('---------- For Regressor %s ---------\n' % regressor_name)
    evaluate_model(predicted_values, y_test)
