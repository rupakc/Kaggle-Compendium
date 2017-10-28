import pandas as pd
import fitter


def distribution_helper(data_list,distribution_list):
    distribution_fit_object = fitter.Fitter(data_list,distributions=distribution_list,timeout=600,verbose=False)
    distribution_fit_object.fit()
    error_map = distribution_fit_object.df_errors.to_dict()
    distribution_error_map = error_map['sumsquare_error']
    return distribution_error_map


def metadata_pipeline(dataframe,distribution_list,filename_to_write):
    summary_frame = dataframe.describe()
    summary_dict = summary_frame.to_dict()
    column_stat_map = dict({})
    for column in summary_frame.columns:
        data_list = list(dataframe[column].values)
        error_map = distribution_helper(data_list,distribution_list)
        statistics_map = summary_dict[column]
        statistics_map.update(error_map)
        column_stat_map[column] = statistics_map
    metadata_frame = pd.DataFrame(column_stat_map)
    metadata_frame.to_csv(filename_to_write)
    return metadata_frame


filename = 'train.csv'
taxi_frame = pd.read_csv(filename)
taxi_frame = taxi_frame.head(1000)
distribution_to_fit = ['expon','exponnorm', 'gausshyper', 'gennorm', 'logistic','uniform','norm','gamma','t']
metadata_pipeline(taxi_frame,distribution_to_fit,'test-meta.csv')

