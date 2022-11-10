"""

    Hecho por Loana Abril Schleich Garcia, entregado el dia 10/11/22.
     ________________________
    | DATOS PARA EL ANALISIS |
    |________________________|
    
    Se seleccionó el archivo "measures.csv".
     _____________________________
    | INTERPRETACIÓN DE LOS DATOS |
    |_____________________________|

    El archivo consta de 15 columnas (desestimando la primera que actúa de índice),
    listadas a continuación de forma ordenada con una breve descripción:
    
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
    como base sus diferentes características físicas.
    
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

from clases import DataFrameManager
from clases import DataCleaner
from clases import TrainAndTest

import seaborn as sns

#Creo objeto para manejar el conjunto de datos.
data_frame = DataFrameManager("../datos/measures.csv", ";")
#Limpio el conjunto de datos.
data_frame.set_data_frame(DataCleaner().fill_nan_values(data_frame.get_data_frame()))

##########################
# GRAFICO DEL PESO MEDIO #
##########################

pesos = data_frame.get_sub_data_frame([4])
mean_w = str(np.round_(np.mean(pesos), decimals = 2)[0])

_, ax = plt.subplots(figsize=(8,6))
ax.hist(pesos, color = "lightcoral")
plt.text(305, 76, s= "\n Peso Medio: " + mean_w +" \n", size=8, ha="left", va="bottom", bbox=dict(boxstyle="square", ec=(1.0, 0.7, 0.5), fc=(1.0, 0.9, 0.8),)).set_bbox({"facecolor":"white", "edgecolor":"black"})

plt.title("Variación del Peso")
plt.show()


"""
     ____________________________
    | ANALISIS DE CORRELATIVIDAD |
    |____________________________|

    La informacion que podemos sacar del siguiente gráfico es que existe una muy alta correlatividad
    positiva entre el peso y las medidas corporales circunferentes, además del porcentaje de grasa corporal.

    La densidad corporal, la edad y la altura por otro lado muestran una correlatividad negativa relativamente mínima.
    
    La densidad y la edad particularmente parecen tener muy poca correlación, pero teniendo en cuenta
    que tenemos una base de datos relativamente escasa no sólo en cuestión de muestras (sujetos), sino de
    características (cada columna representa el 7% de la información), son valores lo suficientemente altos como para
    tener en cuenta.

    Basado en la fundamentación anterior, finalmente:
    
    VARIABLES INDEPENDIENTES:
        Se eligen las columnas 1, 2, 3, y de 5 a 15.

        Dado que se tienen múltiples variables independientes para el análisis, se utilizará un modelo deRegresión Lineal Múltiple.

"""

#############################
# GRAFICO DE CORRELATIVIDAD #
#############################

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


#####################
# GRAFICO DE BARRAS #
#####################

#Comparativa de valores predichos y reales.
mean_abs_err = str(round(metrics.mean_absolute_error(test_vals, pred_vals), 2))
mean_sqr_err = str(round(metrics.mean_squared_error(test_vals, pred_vals), 2))
mean_err = str(np.round_(np.sqrt(metrics.mean_squared_error(test_vals, pred_vals)), decimals = 2))

txt = "Métricas:\n\n"+"Error Absoluto Medio: "+mean_abs_err+"\n"+"Error Cuadratico Medio: "+mean_sqr_err+"\n"+"Raíz del error cuadrático medio: "+mean_err

tt.compare_test_and_predict_vals().plot(kind = "bar", figsize = (10,5), color = ["skyblue","plum"], width = 0.8)
plt.text(35, 9, s= txt, size=8, ha="left", va="bottom", bbox=dict(boxstyle="square", ec=(1.0, 0.7, 0.5), fc=(1.0, 0.9, 0.8),)).set_bbox({"facecolor":"lavenderblush", "edgecolor":"pink"})

plt.title("Valores reales vs. Predicciones")
plt.show()


########################################
# GRÁFICO DE REGRESIÓN LINEAL MÚLTIPLE #
########################################

sns.regplot(x = test_vals, y = pred_vals, scatter_kws = {"color": "palevioletred"}, line_kws = {"color": "black"})

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
    donde las relaciones entre las predicciones y los valores reales tienen una
    gran cercanía con la recta trazada.

    La raíz del error cuadrático medio nos dió un valor de 4.92, que equivale
    a sólo el 2.75% de la media del peso de los sujetos, un valor muy bajo.

    Para finalizar, es posible afirmar que si bien las predicciones no son una garantía, la tasa de aciertos es
    considerablemente muy alta.

"""