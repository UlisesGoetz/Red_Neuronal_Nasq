
# importar las librerÃ­as
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Datos_NAs.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categÃ³ricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Construir la RNA

# Importar Keras y librerÃ­as adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# RNA
clasificador = Sequential() #Objeto RNA

#primera capa oculta
clasificador.add(Dense(units = 7, kernel_initializer = "uniform",  
                     activation = "sigmoid", input_dim = 11))
#input_dim => dimencion del vector de entrada

#segunda capa oculta
clasificador.add(Dense(units = 7, kernel_initializer = "uniform",  activation = "sigmoid"))

# Añadir la capa de salida
clasificador.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
clasificador.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) #Funcion para mejorar la predicidon de nuestras capas ocultas

# Funcion de nuestro objeto donde entrenamos la red neuronal
clasificador.fit(X_train, y_train,  batch_size = 5, epochs = 64)



#predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = clasificador.predict(X_test)

