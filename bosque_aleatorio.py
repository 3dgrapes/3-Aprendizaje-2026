import arboles_numericos as an
import random as ran

def entrenar_bosque(datos, target, clase_default, max_profundidad=3, acc_nodo=1.0, min_ejemplos=0, variables_seleccionadas=None, M=1):
    bosque = []

    for _ in range(M):
        subconjunto = ran.choices(datos, k=len(datos))
        bosque.append(an.entrena_arbol(
            subconjunto, 
            target, 
            clase_default,
            max_profundidad,
            acc_nodo,
            min_ejemplos,
            variables_seleccionadas
            ))

    return bosque






    
