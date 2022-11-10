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
    relevante para el análisis:
    
    VARIABLES DEPENDIENTES:
        Se elige la columna 4.
        El peso es el dato que se desea predecir y cuyo valor suponemos varía en relación al resto.
    
    Se procede a realizar un análisis de correlatividad para seleccionar nuestras variables independientes.
"""

from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd

from clases import DataFrameManager
from clases import DataCleaner
from clases import TrainAndTest

import seaborn as sns

#Creo objeto para manejar el conjunto de datos.
data_frame = DataFrameManager("../datos/measures.csv", ";")
#Limpio el conjunto de datos.
data_frame.set_data_frame(DataCleaner().fill_nan_values(data_frame.get_data_frame()))

"""
 ___________________________
| GRÁFICO DE CORRELATIVIDAD |
|___________________________|

    La informacion que podemos sacar del siguiente gráfico, es que existe una muy alta correlatividad
    positiva entre el peso y las medidas corporales circunferentes, además del porcentaje de grasa corporal.

    El porcentaje de grasa en el cuerpo es lógicamente importante para realizar los cálculos, 
    un sujeto con mayor o menor porcentaje de grasa en su cuerpo va a tender a un
    mayor o menor peso. Un mayor o menor peso a su vez significa un aumento o decremento en las medidas
    físicas debido a los cambios en el volumen corporal.

    La densidad corpora, la edad y la altura por otro lado muestran una correlatividad negativa relativamente mínima.


    Basado en la fundamentación anterior, se concluye:
    
    VARIABLES INDEPENDIENTES:
        Se eligen las columnas 2, 3, y de 5 a 15.
   
    
        
        La edad es un factor importante, el envejecimiento y los cambios hormonales y metabólicos
        que supone favorecen el aumento de peso.
        
        La estatura también es un dato a tener en cuenta, un objeto más alto tiende a ser más pesado debido a su volumen.
        La masa ósea y muscular por sí misma es mayor y aumenta considerablemente su peso, teniendo o no en cuenta
        la cantidad de grasa corporal que pueda poseer.
        
        La densidad, por otro lado, es el cociente entre la masa y el volumen que ocupa, que un cuerpo sea
        muy denso sólo significa que es más compacto que otro con exactamente el mismo peso con menor densidad.

    Dado que se tienen múltiples variables independientes para el análisis, se utilizará un modelo de Regresión Lineal Múltiple.
"""

corr = data_frame.get_sub_data_frame(list(range(1,16))).corr()
corr_per = ((corr["Weight"] * 100).abs().sort_values(ascending = False)).round(decimals = 1)[1:].to_string()

plt.subplots(figsize = (10,5))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = sns.diverging_palette(240, 10, as_cmap=True)) 

plt.text(18, 8, s="Weight (%)\n\n" + corr_per, size=8, ha="left", va="bottom", bbox=dict(boxstyle="square", ec=(1.0, 0.7, 0.5), fc=(1.0, 0.9, 0.8),)).set_bbox({"facecolor":"lavenderblush", "edgecolor":"pink"})

plt.title("Correlatividad")
plt.show()

#################################
# PREPROCESAMIENTO DE LOS DATOS #
#################################

#Creo subconjuntos de variables independientes y dependientes.
independent_vars = data_frame.get_sub_data_frame([1,2,3,5,6,7,8,9,10,11,12,13,14,15])
dependent_vars = data_frame.get_sub_data_frame([4])

#Separo variables dependientes e independientes en conjuntos de prueba y entrenamiento.
tt = TrainAndTest(independent_vars, dependent_vars)
pred_vals = tt.get_predict()
test_vals = tt.get_dependent_test_values()

#Exporto conjuntos de prueba y de entrenamiento.
data_frame.export_data_frame(tt.get_independent_train_values(),"independientes_train")
data_frame.export_data_frame(tt.get_independent_test_values(),"independientes_test")
data_frame.export_data_frame(tt.get_dependent_train_values(),"dependientes_train")
data_frame.export_data_frame(tt.get_dependent_test_values(),"dependientes_test")

######################
# GRAFICO RESULTADOS #
######################


#GRÁFICO DE BARRAS
#Comparativa de valores predecidos y reales.
mean_w = str(np.round_(np.mean(dependent_vars), decimals = 2)[0])
mean_abs_err = str(round(metrics.mean_absolute_error(test_vals, pred_vals), 2))
mean_sqr_err = str(round(metrics.mean_squared_error(test_vals, pred_vals), 2))
mean_err = str(np.round_(np.sqrt(metrics.mean_squared_error(test_vals, pred_vals)), decimals = 2))

txt = "Métricas:\n\n"+"Peso Medio: "+mean_w+"\n"+"Error Absoluto Medio: "+mean_abs_err+"\n"+"Error Cuadratico Medio: "+mean_sqr_err+"\n"+"Raíz del error cuadrático medio: "+mean_err

tt.compare_test_and_predict_vals().plot(kind = "bar", figsize = (10,5), color = ["skyblue","plum"], width = 0.8)
plt.text(35, 9, s= txt, size=8, ha="left", va="bottom", bbox=dict(boxstyle="square", ec=(1.0, 0.7, 0.5), fc=(1.0, 0.9, 0.8),)).set_bbox({"facecolor":"lavenderblush", "edgecolor":"pink"})

plt.title("Valores reales vs. Predicciones")
plt.show()

#GRÁFICO DE REGRESIÓN LINEAL MÚLTIPLE
sns.regplot(x = test_vals, y = pred_vals, scatter_kws = {"color": "crimson"}, line_kws = {"color": "black"})

plt.xlabel("Actual")
plt.ylabel("Predicción")
plt.title("Predicciones en un Modelo RLM")
plt.show()


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