import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""
Clase para manejar el archivo csv.
"""
class dataFrameManager:
    def __init__(self):
        self.dF = pd.read_csv("../datos/BaseUnificadaEstaciones.csv")
        self.__independentValues = self.dF.iloc[:,1:-1].values
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
    def transCategoricalData(self, vector_vals):
        return LabelEncoder().fit_transform(vector_vals)