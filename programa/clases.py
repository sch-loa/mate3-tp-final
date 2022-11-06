import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

"""
Clase para manejar el archivo csv.
"""
class DataFrameManager:

    def __init__(self, _path):
        self.__data_frame = pd.read_csv(_path)
        self.__independent_variables = np.array([])
        self.__dependent_variables = np.array([])
    
    def get_data_frame(self):
        return self.__data_frame
    
    def set_data_frame(self, data_frame):
        self.__data_frame = data_frame

    def get_independent_variables(self):
        if(self.__independent_variables.size == 0):
            raise Exception("Las variables independientes no fueron inicializadas")
        return self.__independent_variables
    
    def get_dependent_variables(self):
        if(self.__dependent_variables.size == 0):
            raise Exception("Las variables dependientes no fueron inicializadas")
        return self.__dependent_variables
    
    def set_independent_variables(self, list_col_names):
        self.__independent_variables = self.__data_frame.loc[:,list_col_names]
    
    def set_dependent_variables(self, list_col_names):
        self.__dependent_variables = self.__data_frame.loc[:,list_col_names]


class DataCleaner:

    def transform_categorical_data(self, data_frame, list_col_names):
        return ce.OneHotEncoder(cols = list_col_names).fit_transform(data_frame)

    def fill_nan_values(self, data_frame, list_col_names):
        simple_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
        data_frame.loc[:,list_col_names] = simple_imputer.fit_transform(data_frame.loc[:,list_col_names])
        return data_frame

class TrainAndTest:
    def __init__(self, independent_vars_frame, dependent_vars_frame):
        self.__ind_train, self.__ind_test, self.__dep_train, self.__dep_test = train_test_split(independent_vars_frame, dependent_vars_frame, test_size = 0.2, random_state = 9)
    
    def get_indepedent_train_values():
        return self.__ind_train

    def get_indepedent_test_values():
        return self.__ind_test
    
    def get_depedent_train_values():
        return self.__dep_train

    def get_depedent_test_values():
        return self.__dep_test
    


        
