import arboles_numericos as an
import random as ran
from collections import Counter

def entrenar_bosque(datos, target, clase_default, max_profundidad=3, acc_nodo=1.0, min_ejemplos=0, variables_seleccionadas=None, M=1):
    bosque = [] # Pone todos los árboles (o subconjuntos) dento de un "bosque"

    for _ in range(M): # Por cada repetición
        subconjunto = ran.choices(datos, k=len(datos)) # ...creamos un subconjunto de datos
        bosque.append(an.entrena_arbol( # ...entrenamos un arbol usando el subconjunto y lo agregamos a nuestro bosque
            subconjunto, 
            target, 
            clase_default,
            max_profundidad,
            acc_nodo,
            min_ejemplos,
            variables_seleccionadas
            ))

    return bosque # Cuando termianmos con nuestros loops, devolvemos todo el bosque

def predecir_bosque(bosque, datos):
    predicciones_finales = [] # Aquí guardamos las predicciones finales hechas por cada árbol

    for dato in datos: # por cada dato en nuestra lista
        predicciones = [] # Vacía nuestra lista de predicciones hechas

        for arbol in bosque:
            predicciones.append(arbol.predice(dato))

        predicciones_finales.append(Counter(predicciones).most_common(1)[0][0])

    return predicciones_finales

def evalua_bosque(bosque, datos, target): # Esta es una función auxiliar que hacemos para evaluar errores con bosques en lugar de arboles
    predicciones = predecir_bosque(bosque, datos) # Hacemos nuestras predicciones con el bosque.

    # Contamos los erroes
    errores = 0
    
    # ...por cada dato que tenemos
    for index in range(len(datos)):
        if predicciones[index] != datos[index][target]: # Si la predicción en el indice no es la misma que el valor que se encuentra en el la clase localizada en el indice dentro de nuestra base de datos
            errores += 1 # ...contamos un error

    return errores/len(datos) # Devolvemos el "porcentaje" de la probablidad de que cometamos un error.