"""
Hecho por Loana Abril Schleich Garcia, entregado el dia xx/11/22.
 ________________________
| DATOS PARA EL ANALISIS |
|________________________|
    
    Se seleccionó el archivo "measures.csv".
 _____________________________
| INTERPRETACIÓN DE LOS DATOS |
|_____________________________|

    El archivo consta de 16 columnas; una actuando de índice y
    15 columnas con datos de distintos sujetos relacionados
    a su corporalidad, listados a continuación de forma ordenada:
    
    densidad corporal, grasa corporal, edad, peso, estatura, cuello,
    pecho, abdomen, cadera, muslo, rodilla, tobillo, bíceps,
    antebrazo, muñeca.
    
    Las últimas 10 columnas anteriores corresponden a medidas corporales.
    
    Las unidades son ignorables para la realización de los cálculos, pues se asume que
    se usan las mismas para cada columna.
    
 ________________________
| OBJETIVOS Y ESTRATEGIA |
|________________________|

    Mediante el procesamiento del conjunto de datos seleccionado,
    se tiene como propósito producir predicciones en relación a la edad
    de las personas, teniendo como base sus distintas características físicas.
    
    En primera instancia, se deberá seleccionar cuidadosamente la información
    relevante para el análisis: se toma la decisión de descartar la densidad
    y la grasa corporal.
    La primera hace referencia a la cantidad de masa
    corporal libre de grasa en el cuerpo, mientras que la segunda es lo contrario.
    
    La información que puede sacarse de esos datos es únicamente si el sujeto en cuestión
    se encuentra en un estado físico óptimo en relación al peso. Dado que es una caractaristica
    alterable, que puede oscilar en cortos periodos de tiempo debido a distintos factores
    (ambientales, alimenticios, genéticos, etc) ambos son datos que realmente no aportan
    información nueva ni relevante al objetivo.

    Por otro lado, las medidas físicas sí resultan necesarias: Si bien varían de sujeto a sujeto
    pudiendo haber tendencias producto de factores externos, son caracteristicas que se desarrollan
    a medida que un ser vivo crece. Entonces es posible definir una tendencia dentro de un rango etario.
"""

from clases import DataFrameManager
from clases import DataCleaner
from clases import TrainAndTest

#Creo objeto para manejar el conjunto de datos
data_frame = DataFrameManager("../datos/measures.csv", ";")

#Creo subconjuntos de variables independientes y dependientes.
independent_vars = data_frame.get_sub_data_frame(list(range(4,16)))
dependent_vars = data_frame.get_sub_data_frame([3])

#Limpio datos numéricos en caso de que hayan valores nulos
independent_vars = DataCleaner().fill_nan_values(independent_vars)
dependent_vars = DataCleaner().fill_nan_values(dependent_vars)

print(data_frame.get_data_frame())
print(dependent_vars)
print(independent_vars)


tt = TrainAndTest(independent_vars, dependent_vars)

print(tt.scale_data(tt.get_indepedent_train_values()))