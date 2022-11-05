import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""
Clase para manejar el archivo csv.
Requisitos implicitos para su utilización:
    A. Existe una única variable dependiente, y se encuentra en la última columna.
    B. Existe una cantidad indefinida de variables independientes, se toman todas las columnas menos la última.
"""
class dataFrameManager:
    def __init__(self, _path):
        self.dF = pd.read_csv(_path)
        self.__independentValues = self.dF.iloc[:,:-1].values
        self.__dependentValues = self.dF.iloc[:, -1].values
    
    def getIndependentValues(self):
        return self.__independentValues
    
    def getDependentValues(self):
        return self.__dependentValues
    
    def setIndependentValues(self, matrix_vals):
        self.__independentValues = matrix_vals
    
    def setDependentValues(self, vector_vals):
        self.__dependentValues = vector_vals


class dataCleaner:
    