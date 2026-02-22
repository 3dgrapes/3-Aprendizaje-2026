import utileria as ut
import bosque_aleatorio as ba
import os
import random

# El trabajo lo hice basandome de las pruebas proveidas por usted y algunos detalles con la ayuda de IA.  
# Voy a tratar de explicar que pasa personalmente en la lineas de codigo para ver si lo entiendo. 

# Aquí saco los datos de la base de datos
url =  "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip" # Tomas el url de donde se descarga el zip
archivo = "datos/banknote.zip" # El archivo a buscar
archivo_datos = "datos/data_banknote_authentication.txt" # Los datos a leer dentro del archivo
atributos = ['variance', 'skewness', 'curtosis', 'entropy', 'class'] # Atributos 

# Me aseguro que existe alguna parte para poner mis datos
if not os.path.exists("datos"): # El directorio datos no existe?
    os.makedirs("datos") # Haz el directorio
if not os.path.exists(archivo): # El arhivo no existe? 
    ut.descarga_datos(url, archivo) # Descarga los datos...
    ut.descomprime_zip(archivo) # y los descomprimimos

# Extraigo datos
datos = ut.lee_csv( # Leemos la base de datos
    archivo_datos,  # El archivo a leer 
    atributos=atributos, # Los atributos a buscar 
    separador="," # Lo que separa cada valor
)

# Limpio y convierto los datos a númericos
for d in datos: # Para cada fila de nuestra base de datos...
    d['class'] = int(d['class']) 
    d['variance'] = float(d['variance'])
    d['skewness'] = float(d['skewness'])
    d['curtosis'] = float(d['curtosis'])
    d['entropy'] = float(d['entropy'])

# Seleccionamos nuestro target
target = 'class' 

# La mezcla y separación (80/20) se queda idéntica
random.seed(42) # Ponemos un seed 
random.shuffle(datos) # Mesclamos nuestros datos
N = int(0.8 * len(datos)) # Tomamos una cantidad de datos (no todos al parecer)
datos_entrenamiento = datos[:N] # ... y tomamos la primera parte para que sean de entrenamiento
datos_validacion = datos[N:] # y tomamos el resto para que sean de validación



# 1. Aumentamos el numero de árboles en un bosque

# Calculamos errores con diferentes profundidades
errores_M  = []
for cantidad in [1, 5, 10, 50, 100]:

    # Entrenamos un bosque 
    bosque = ba.entrenar_bosque(
            datos=datos_entrenamiento, 
            target=target, 
            clase_default=0, 
            max_profundidad=5, 
            variables_seleccionadas=2, 
            M=cantidad # Modificamos la cantidad de árboles
        )
    
    # Calificamos nuestro bosque
    error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)

    # Guardamos los resultados
    errores_M.append((cantidad, error_en_muestra, error_en_validacion))

# E imprimimos los resultados
print('\nCambiando numero de árboles')
print('Arboles'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for cantidad, error_entrenamiento, error_validacion in errores_M:
    print(
        f'{cantidad}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

# En los siguientes experimentos reusamos el codigo, pero cambiamos los parametros para que se vean reflejados los cambios que queremos observar


# 2. Aumentamos la profundidad 

print('\nCambiando la Profundidad Máxima')
print('Profundidad'.center(15) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 45)

errores_prof = []
for profundidad in [1, 3, 5, 10, 20]:
    
    bosque = ba.entrenar_bosque(
        datos=datos_entrenamiento, 
        target=target, 
        clase_default=0, 
        max_profundidad=profundidad, # Modificamos la profundidad
        variables_seleccionadas=2,   
        M=50                         
    )
    
    error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)
    
    errores_prof.append((profundidad, error_en_muestra, error_en_validacion))

    print(
        f'{profundidad}'.center(15) 
        + f'{error_en_muestra:.4f}'.center(15) 
        + f'{error_en_validacion:.4f}'.center(15)
    )
print('-' * 45)


# 3. Cambiamos la cantidad de variables


print('\nCambiando Variables Seleccionadas en cada Nodo')
print('Variables'.center(15) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 45)

errores_vars = []
for vars_sel in [1, 2, 3, 4]:
    
    bosque = ba.entrenar_bosque(
        datos=datos_entrenamiento, 
        target=target, 
        clase_default=0, 
        max_profundidad=5,           
        variables_seleccionadas=vars_sel, # Modificamos el numero de variables
        M=50                         
    )
    
    error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)
    
    errores_vars.append((vars_sel, error_en_muestra, error_en_validacion))
    
    print(
        f'{vars_sel}'.center(15) 
        + f'{error_en_muestra:.4f}'.center(15) 
        + f'{error_en_validacion:.4f}'.center(15)
    )
print('-' * 45)

# Conclusiones de los cambios

# 1. Número de árboles: Ein se hacienta mucho más rápido que Eout, pero ambas se quedan alrededor del mismo error cuando aumentan los árboles.
# 2. Profundidad máxima: Ein rapidamente se hacienta y se acerca mucho a 0. Eout disminuye rapidamente bajo 1, pero llega a un momento en donde batalla en bajar (¡incluso sube algunas veces!).
# 3. Variables seleccionadas: Ein y Eout inician muy cerca del 0. Ambos aumentan y disminuyen en cierto punto pero, llega un momento en donde disminuyen apropiadamente.