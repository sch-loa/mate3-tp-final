import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

"""
Clase para manejar el archivo csv.
"""
class DataFrameManager:
    def __init__(self, _path, _delimiter):
        self.__data_frame = pd.read_csv(_path, sep = _delimiter)
    
    def get_data_frame(self):
        return self.__data_frame
    
    def get_sub_data_frame(self, list_of_colums):
        return self.__data_frame.iloc[:,list_of_colums]

class DataCleaner:
    def fill_nan_values(self, data_frame):
        #try:
        simple_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
        data_frame = simple_imputer.fit_transform(data_frame)
        #except:
            #print("El conjunto posee columnas con datos no numéricos.")

        return data_frame

    def scale_data(self, array_vars):
        return StandardScaler().fit_transform(array_vars)

class TrainAndTest:
    def __init__(self, independent_vars, dependent_vars):       
        self.__ind_train, self.__ind_test, self.__dep_train, self.__dep_test = train_test_split(independent_vars, dependent_vars, test_size = 0.2, random_state = 10)

    def get_independent_train_values(self):
        return self.__ind_train

    def get_independent_test_values(self):
        return self.__ind_test
    
    def get_dependent_train_values(self):
        return self.__dep_train

    def get_dependent_test_values(self):
        return self.__dep_test

    def scale_all_data(self):
        self.__ind_train = StandardScaler().fit_transform(self.__ind_train)
        self.__ind_test = StandardScaler().fit_transform(self.__ind_test)
        self.__dep_train = StandardScaler().fit_transform(self.__dep_train)
        self.__dep_test = StandardScaler().fit_transform(self.__dep_test)
        
    def get_predict(self):
        regression = LinearRegression().fit(self.__ind_train, self.__dep_train)
        return regression.predict(self.__ind_test)
    
    def compare_train_and_test(self):
        return pd.DataFrame({"Actual": np.reshape(self.__dep_test.flatten(), -1), "Prediccion": np.reshape(self.get_predict(), -1)})
    