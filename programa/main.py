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
    
    density: densidad corporal, relación entre la masa de un cuerpo y el volumen que ocupa.
    fat: grasa corporal
    age: edad
    weight: peso del sujeto
    height: altura del sujeto
    neck: medida de la circunferencia del cuello
    chest: ídem anterior, del pecho
    abdomen: ídem anterior, del abdomen
    hip: ídem anterior, pero de la cadera
    thigh: ídem anterior, pero del muslo
    knee: ídem anterior, pero de la rodilla
    ankle: ídem anterior, del tobillo.
    bicep: ídem, bícep
    forearm: ídem, antebrazo
    wrist: ídem, muñeca
    
    Las unidades son ignorables para la realización de los cálculos, pues se asume que
    se usan las mismas para cada columna.
    
 ________________________
| OBJETIVOS Y ESTRATEGIA |
|________________________|

    Mediante el procesamiento del conjunto de datos seleccionado,
    se tiene como propósito predecir el peso de las personas, teniendo
    como base sus distintas características físicas.
    
    En primera instancia, se deberá seleccionar cuidadosamente la información
    relevante para el análisis: se toma la decisión .

"""
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np

from clases import DataFrameManager
from clases import DataCleaner
from clases import TrainAndTest

import seaborn

from sklearn.metrics import r2_score

#################################
# PREPROCESAMIENTO DE LOS DATOS #
#################################

#Creo objeto para manejar el conjunto de datos
data_frame = DataFrameManager("../datos/measures.csv", ";")

#Creo subconjuntos de variables independientes y dependientes.
independent_vars = data_frame.get_sub_data_frame([1,2,3,5,6,7,8,9,10,11,12,13,14,15])
dependent_vars = data_frame.get_sub_data_frame([4])

#Limpio datos numéricos en caso de que hayan valores nulos
independent_vars = DataCleaner().fill_nan_values(independent_vars)
dependent_vars = DataCleaner().fill_nan_values(dependent_vars)

tt = TrainAndTest(independent_vars, dependent_vars)
#tt.scale_all_data()
pred_vals = tt.get_predict()
test_vals = tt.get_dependent_test_values()

print(f"{tt.compare_test_and_predict()} \n")

######################
# GRAFICO RESULTADOS #
######################

#Creo gráfico de barras para comparar las variables de prueba y las predicciones para visualizar la precisión.
tt.compare_test_and_predict().plot(kind = "bar", figsize = (10,5), color = ["blue","mediumpurple"])

plt.title("Diferencia entre los valores reales y los de la predicción")
plt.show()

#Creo gráfico de regresión lineal
seaborn.regplot(x = test_vals, y = pred_vals, scatter_kws = {"color": "mediumpurple"}, line_kws = {"color": "black"})
plt.xlabel("Actual")
plt.ylabel("Predicción")
plt.title("Predicciones en un modelo de RLM")
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