import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

##########################################
# Manejador del archivo de extensión csv #
##########################################

class DataFrameManager:
    #Lee un archivo csv de ruta _path, separando las columnas con _delimitador.
    def __init__(self, _path, _delimiter):
        self.__data_frame = pd.read_csv(_path, sep = _delimiter)
    
    #Retorna el DataFrame
    def get_data_frame(self):
        return self.__data_frame
    
    #Cambia el DataFrame actual por el conjunto pasado como parámetro.
    def set_data_frame(self, data_frame):
        self.__data_frame = pd.DataFrame(data_frame)

    #Retorna todas las filas de una partición del DataFrame,
    #pasando como parámetro una lista con los índices de las columnas.
    def get_sub_data_frame(self, list_of_colums):
        return self.__data_frame.iloc[:,list_of_colums]
    
    #Exporta un conjunto de datos en formato de archivo csv.
    def export_data_frame(self, data_set, name):
        if(not isinstance(data_set, pd.DataFrame)):
            data_set = pd.DataFrame(data_set)
        data_set.to_csv("./" + name + ".csv", encoding = "utf-8")


########################################
# Limpiador de los datos del DataFrame #
########################################

class DataCleaner:
    #Retorna el DataFrame pasado como parámetro, modificando las filas
    #que tienen valores NaN con el método del valor medio.
    def fill_nan_values(self, data_frame):
        if(data_frame.isnull().values.any()):
            simple_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
            data_frame = simple_imputer.fit_transform(data_frame)
        return data_frame


#####################################################
# Manejador de variables de entrenamiento y pruebas #
#####################################################

class TrainAndTest:
    #Particiona la matriz de variables independientes y el vector de variables
    #dependientes en sus correspondientes conjuntos de entrenamiento y prueba.
    def __init__(self, independent_vars, dependent_vars):       
        self.__ind_train, self.__ind_test, self.__dep_train, self.__dep_test = train_test_split(independent_vars.values, dependent_vars.values, test_size = 0.2, random_state = 0)
    
    #Retorna el conjunto de entrenamiento de las variables independientes.
    def get_independent_train_values(self):
        return self.__ind_train
    
    #Retorna el conjunto de prueba de las variables independientes.
    def get_independent_test_values(self):
        return self.__ind_test
    
    #Retorna el conjunto de entrenamiento de las variables dependientes.
    def get_dependent_train_values(self):
        return self.__dep_train

    #Retorna el conjunto de prueba de las variables dependientes.
    def get_dependent_test_values(self):
        return self.__dep_test
   
    #Retorna la predicción obtenida a partir de los conjuntos de prueba.
    def get_predict(self):
        regression = LinearRegression().fit(self.__ind_train, self.__dep_train)
        return regression.predict(self.__ind_test)
    
    #Retorna un DataFrame con los datos de prueba y los datos obtenidos
    #a partir de las predicciones como columnas.
    def compare_test_and_predict_vals(self):
        return pd.DataFrame({"Actual": np.reshape(self.__dep_test.flatten(), -1), "Prediccion": np.reshape(np.round_(self.get_predict(), decimals = 2), -1)})
    