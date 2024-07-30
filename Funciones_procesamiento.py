# Funciones de procesamiento y preparación del modelo

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor

#################################################################################################################
#################################################################################################################

# Esta función nos da una agrupación de marcas que suelen tener precios más altos

def marcas_lujo(df):
    marcas_de_lujo = [
    'MERCE', 'BMW', 'AUDI', 'PORSCHE', 'JAGUAR', 'LEXUS', 'TESLA', 
    'MASERATI', 'ASTON MARTIN', 'CADILLAC', 'INFINITI', 'ALFA ROMEO' #'ROVER'
    ]
    # Crear la columna 'es_de_lujo'
    df['Lujo'] = df['Marca'].isin(marcas_de_lujo)

    return df

#################################################################################################################
#################################################################################################################

# Con esta función, recorremos los comentarios, creamos una columna llamada 'Averia' en la cual, si se detectan palabras clave relacionadas 
# con averías o similiar pone un 1, si no, un 0.

def busca_averia(df):
    # Inicializar la columna 'Palabras clave' en 0
    df['Averia'] = 0
    
    # Lista de palabras clave
    lista_palabras = ['aver','roto',"no funciona","para piezas","desperfecto","fallo","rotura","error","defect","malfuncionamiento","sinies","sinestr"]  # Puedes añadir más palabras clave aquí
    
    # Definir una función para buscar palabras clave en los comentarios
    def contiene_palabras_clave(comentario):
        comentario = str(comentario).lower()
        for palabra in lista_palabras:
            if palabra in comentario:
                return 1
            
        return 0
    
    # Aplicar la función solo a las filas donde 'Tipo de vehiculo' es 'Furgoneta'
    df.loc[df['Tipo de vehiculo'] != 'asd', 'Averia'] = df[df['Tipo de vehiculo'] != 'asd']['Comentarios'].apply(contiene_palabras_clave)
    
    return df  

#################################################################################################################
#################################################################################################################

# Con esta función, recorremos los comentarios de las furgonetas, creamos una columna llamada 'Camper' en la cual, si se detectan palabras clave relacionadas 
# con camperización o similiar pone un 1, si no, un 0.

def busca_camper(df):
    # Inicializar la columna 'Palabras clave' en 0
    df['Camper'] = 0
    
    # Lista de palabras clave
    lista_palabras = ['camper', 'casa']  # Puedes añadir más palabras clave aquí
    
    # Definir una función para buscar palabras clave en los comentarios
    def contiene_camper(comentario):
        comentario = str(comentario).lower()
        for palabra in lista_palabras:
            if palabra in comentario:
                return 1
        return 0
    
    # Aplicar la función solo a las filas donde 'Tipo de vehiculo' es 'Furgoneta'
    df.loc[df['Tipo de vehiculo'] == 'Furgoneta', 'Camper'] = df[df['Tipo de vehiculo'] == 'Furgoneta']['Comentarios'].apply(contiene_camper)
    
    return df

#################################################################################################################
#################################################################################################################

# Esta función se encarga de preparaar los dummies para hacer el modelo

# TRAIN-TEST SPLIT
# Previamente hay que hacer la división train-test de la siguiente manera (para poder usar las siguientes funciones):

# X=df.drop(columns=['Modelo','Comentarios', 'Nombre usuario', 'Puntuacion', 'Nº Ventas', 'Ubicacion','Intervalo','Antiguedad', 'Precio'])
# y=np.log(df['Precio'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # random_state=42
# xtrain_normal=X_train.copy()
# xtest_normal=X_test.copy()

# Se llama a la función con el siguiente código:
# X_train, X_test = pre_model(X_train, X_test)

def pre_model(X_train, X_test):
    
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    
    # Dummies con OneHotEncoder   
    columnas_cat=['Marca', 'Tipo de vehiculo', 'Combustible', 'Cambio']  #'Antigüedad', 'Potencia', 'Año', 'Plazas', 'Nº de puertas',
    objeto_dummies = OneHotEncoder(sparse_output=False, #por defecto el formato el formato es sparse
                                drop='first', # no creamos una nueva columna con la primera categoria que encuentra
                                handle_unknown='ignore') # ignora nuevas categorias en el test
    
    objeto_dummies.fit(X_train[columnas_cat])
    objeto_dummies.transform(X_train[columnas_cat])

    nombre_columnas= objeto_dummies.get_feature_names_out(columnas_cat)

    X_train[nombre_columnas] = objeto_dummies.transform(X_train[columnas_cat])
    X_train = X_train.drop(columns=columnas_cat)

    X_test[nombre_columnas] = objeto_dummies.transform(X_test[columnas_cat])
    X_test=X_test.drop(columns=columnas_cat)
    
    return X_train, X_test

#################################################################################################################
#################################################################################################################

# Una vez lanzado el modelo, podemos analizar el error usando esta función. La función nos devuelve el DataFrame 
# original (sin columnas transformadas), y añade 3 columnas más: Predicción (el resultado del modelo), Error (diferencia
# entre la predicción y el precio del anuncio) y AbsError (Error en valor absoluto, para poder ver más fácilmente las filas con más error,
# tanto en positivo como en negativo).
# se llama a la función de la siguiente manera:

# df_final=ver_error(xtrain_normal,xtest_normal,y_train,y_test,yhat_train,yhat_test)

# Para ver los valores ordenados de mayor a menor por error absoluto, se puede usar el siguiente código:

# df_final_sorted = df_final.sort_values(by='AbsError',ascending=False)

def ver_error(xtrain_normal,xtest_normal,y_train,y_test,yhat_train,yhat_test):
    import numpy as np
    df_train = pd.concat([xtrain_normal, np.exp(y_train)], axis=1)
    df_test = pd.concat([xtest_normal, np.exp(y_test)],axis=1)
    df_train['Prediccion']=np.exp(yhat_train)
    df_test['Prediccion']=np.exp(yhat_test)
    df_final=pd.concat([df_train,df_test],axis=0)
    df_final['Error']=df_final['Prediccion']-df_final['Precio']
    df_final['AbsError'] = df_final['Error'].abs()
    
    return df_final

#################################################################################################################
#################################################################################################################

# Esta función serviría para no tener que hacer el train-test split, en caso de que quieras poner el modelo en produccion

def produccion(df):
    import pandas as pd
    import numpy as np
    import joblib
    
    X=df.drop(columns=['Modelo', 'Comentarios', 'Nombre usuario', 'Puntuacion',
                   'Nº Ventas','Ubicacion','latitud','longitud', 'Precio'])
    #y=np.log(df['Precio'])
    columnas_cat=['Marca', 'Tipo de vehiculo', 'Combustible', 'Cambio']
    encoder = joblib.load('encoder_entrenado.joblib')

    columnas_cat=['Marca', 'Tipo de vehiculo', 'Combustible', 'Cambio']
    nombre_columnas= encoder.get_feature_names_out(columnas_cat)

    X[nombre_columnas] = encoder.transform(X[columnas_cat])
    X=X.drop(columns=columnas_cat)
    X['Camper']=X['Camper'].astype('int')

    return X #, y

#################################################################################################################
#################################################################################################################

# esta función incorpora el modelo pre-entrenado.
# es importante que el archivo 'modelo_entrenado.joblib' esté en la misma carpeta que todos los demás

def predecir(X):
    import joblib
    modelo = joblib.load('modelo_entrenado.joblib')

    yhat=modelo.predict(X)
    pred=pd.DataFrame(np.exp(yhat))
    return pred

#################################################################################################################
#################################################################################################################