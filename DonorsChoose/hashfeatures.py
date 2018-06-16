import numpy as np
from sklearn.feature_extraction import FeatureHasher
import preprocess


class FeatureHash:
    
    def __init__(self,max_feature_num=400,input_data_type='string'):
        self.feature_hash = FeatureHasher(n_features=max_feature_num,input_type=input_data_type)
    
    def get_feature_set(self,train_data):
        return self.feature_hash.transform(preprocess.tokenize_string_list(train_data," ")).todense()
    
    def fit_feature_model(self,train_data):
        self.feature_hash.fit(preprocess.tokenize_string_list(train_data," "))
    
    def get_feature_model(self):
        return self.feature_hash