"""
Hecho por Loana Abril Schleich Garcia, entregado el dia xx/11/22.
 ________________________
| DATOS PARA EL ANALISIS |
|________________________|
    
    Se seleccionó el archivo "BaseUnificadaEstaciones.csv",
    de la fuente "https://data.buenosaires.gob.ar/dataset/subte-viajes-molinetes",
    recurso encontrable en la sección "Recursos del dataset" como
    "Cantidad de pasajeros por estación cada 15 minutos del año 2020".

    De los datos proporcionados se contempla la cantidad de pasajeros de subte
    por línea y por estación en una fecha, hora y rango horario determinado.
    
    Distinción entre hora y rango horario: 
        Un rango horario comprende un conjunto de horas (véase columna "DESDE" y "HORA" respectivamente).
        Ejemplo: Las horas 8:00, 8:15, 8:30 y 8:45 AM exclusivamente pertenecen al rango de las 8 AM.

    El objetivo del análisis es predecir la cantidad de pasajeros en cada línea,
    a continuación identificamos si existen datos no esenciales para la realización de los cálculos:
    
    
    
   

"""
from clases import DataFrameManager
from clases import DataCleaner

#dfFile = DataFrameManager("../datos/BaseUnificadaEstaciones.csv")
dfFile = DataFrameManager("../datos/Datos.csv")
cleaner = DataCleaner()
dfFile.set_dependent_variables(["Compro"])
dfFile.set_independent_variables(["Pais","Edad","Salario"])

print(dfFile.get_independent_variables())
print("aaaaa")
print(cleaner.transform_categorical_data(dfFile.get_independent_variables(),["Pais"]))
print(cleaner.fill_nan_values(dfFile.get_independent_variables(), ["Edad","Salario"]))