"""
Hecho por Loana Abril Schleich Garcia, entregado el dia xx/11/22.
 ________________________
| DATOS PARA EL ANALISIS |
|________________________|
    
    Se seleccionó el archivo "measures.csv".
 _____________________________
| INTERPRETACIÓN DE LOS DATOS |
|_____________________________|

    El archivo consta de 15 columnas (desestimando la primera que actúa de índice),
    listadas a continuación de forma ordenadacon una nreve descripción:
    
    1. density: Densidad corporal, relación entre la masa de un cuerpo y el volumen que ocupa.
    2. fat: Grasa corporal en la totalidad de la masa.
    3. age: Edad
    4. weight: Peso
    5. height: Estatura
    
    Columnas con las medidas de la circunferencia de:
    
    6. neck: Cuello
    7. chest: Pecho
    8. abdomen: Abdomen
    9. hip: Cadera
    10. thigh: Muslo
    11. knee: Rodilla
    12. ankle: Tobillo
    13. bicep: Bícep
    14. forearm: Antebrazo
    15. wrist: Muñeca
    
    Las unidades son ignorables para la realización de los cálculos, pues se asume que
    se usan las mismas para cada columna.
    
 ________________________
| OBJETIVOS Y ESTRATEGIA |
|________________________|

    Mediante el procesamiento del conjunto de datos seleccionado,
    se tiene como propósito predecir el peso de las personas, teniendo
    como base sus distintas características físicas.
    
    En primera instancia, se deberá seleccionar cuidadosamente la información
    relevante para el análisis: a continuaciónse listan las columnas
    seleccionadas y su fundamentación:
    
    VARIABLES DEPENDIENTES:
        Se elige la columna 4.
        El peso es el dato que se desea predecir y cuyo valor suponemos varía en relación al resto.

    VARIABLES INDEPENDIENTES:
        Se eligen las columnas 2, 3, y de 5 a 15.
   
        El peso está fuertemente ligado a la cantidad de masa de la que se compone el cuerpo,
        ya sea grasa, músculo o hueso. El porcentaje de grasa en el cuerpo es esencial para realizar
        los cálculos, un sujeto con mayor o menor porcentaje de grasa en su cuerpo va a tender a un
        mayor o menor peso. Un mayor o menor peso a su vez significa un aumento o decremento en las medidas
        físicas debido a los cambios en el volumen corporal.
        
        La edad es un factor importante, el envejecimiento y los cambios hormonales y metabólicos
        que supone favorecen el aumento de peso.
        
        La estatura también es un dato a tener en cuenta, un objeto más alto tiende a ser más pesado debido a su volumen.
        La masa ósea y muscular por sí misma es mayor y aumenta considerablemente su peso, teniendo o no en cuenta
        la cantidad de grasa corporal que pueda poseer.
        
        La densidad, por otro lado, es el cociente entre la masa y el volumen que ocupa, que un cuerpo sea
        muy denso sólo significa que es más compacto que otro con exactamente el mismo peso con menor densidad.

    Dado que se tienen múltiples variables independientes para el análisis, se utilizará un modelo de Regresión Lineal Múltiple.
"""

from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np

from clases import DataFrameManager
from clases import DataCleaner
from clases import TrainAndTest

import seaborn

#################################
# PREPROCESAMIENTO DE LOS DATOS #
#################################

#Creo objeto para manejar el conjunto de datos
data_frame = DataFrameManager("../datos/measures.csv", ";")

#Creo subconjuntos de variables independientes y dependientes.
independent_vars = data_frame.get_sub_data_frame([2,3,5,6,7,8,9,10,11,12,13,14,15])
dependent_vars = data_frame.get_sub_data_frame([4])

#Limpio datos numéricos en caso de que hayan valores nulos
independent_vars = DataCleaner().fill_nan_values(independent_vars)
dependent_vars = DataCleaner().fill_nan_values(dependent_vars)

tt = TrainAndTest(independent_vars, dependent_vars)
"""
#tt.scale_all_data()
"""
pred_vals = tt.get_predict()
test_vals = tt.get_dependent_test_values()

#print(f"{tt.compare_test_and_predict()} \n")

######################
# GRAFICO RESULTADOS #
######################

#Creo gráfico de barras para comparar las variables de prueba y
#las predicciones para visualizar la precisión.
tt.compare_test_and_predict().plot(kind = "bar", figsize = (10,5), color = ["skyblue","plum"], width = 0.8)
plt.title("Valores reales vs. Predicciones")
plt.show()

#Creo gráfico de regresión lineal
seaborn.regplot(x = test_vals, y = pred_vals, scatter_kws = {"color": "plum"}, line_kws = {"color": "black"})
plt.xlabel("Actual")
plt.ylabel("Predicción")
plt.title("Predicciones en un modelo RLM")
plt.show()

print("Media del peso de los sujetos:", np.round_(np.mean(dependent_vars), decimals = 2), "\n")
print("Error Absoluto Medio:", round(metrics.mean_absolute_error(test_vals, pred_vals), 2))
print("Error Cuadratico Medio:", round(metrics.mean_squared_error(test_vals, pred_vals), 2))
print("Raíz del error cuadrático medio:", np.round_(np.sqrt(metrics.mean_squared_error(test_vals, pred_vals)), decimals = 2))

"""
 ______________
| CONCLUSIONES |
|______________|

El gráfico de barras a simple vista mostró una gran cercanía entre
los valores reales y las predicciones. No fue distinto con el gráfico de regresión lineal,
donde las distintas relaciones entre las predicciones y los valores reales tienen una
gran cercanía con la recta trazada.

La raíz del error cuadrático medio nos dió un valor de 4.92, que equivale
a sólo el 2.92% de la media del peso de los sujetos, un valor muy bajo. Esto equivale
a una presición considerablemente alta.

Se concluye que hubo una buena correlación entre los datos utilizados para la predicción  y el valor a predecir.
"""