import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")

# Eliminar columnas con más del 70% de valores faltantes
umbral = 0.7
columnas_a_eliminar = df.columns[df.isnull().mean() > umbral]
df.drop(columns=columnas_a_eliminar, inplace=True)
print(columnas_a_eliminar)

#Llenado de datos faltantes de tipo float64 con la mediana
for col in df.select_dtypes(include=['float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

#Llenado de datos faltantes de tipo str con la moda
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

#Visualizacion de Datos faltantes
datos_faltantes =df.isnull().sum()
print(datos_faltantes)


#-----------------------Procesamiento de variables categoricas---------------#
#----------------------------------------------------------------------------#


#Codificacion para variables binarias 
from sklearn.calibration import LabelEncoder

columnas_binarias = ['bin_3','bin_4']

#Codificacion para bin_3
le = LabelEncoder()
label_encoder_bin3 = df['bin_3'].values
bin_3_np = le.fit_transform(label_encoder_bin3)

#Codificacion para bin_4
le = LabelEncoder()
label_encoder_bin4 = df['bin_4'].values
bin_4_np = le.fit_transform(label_encoder_bin4)

#Creacion de dataframe para las variables binarias codificadas
d = {
    'bin_3' : bin_3_np,
    'bin_4' : bin_4_np
}

#Se agregan los dos columnas a un dataframe
X_label_endoded_df = pd.DataFrame(data = d)

#Visualizacion de las variables binarias codificadas
print(X_label_endoded_df.head()) 


#----------------------------------------------------------------------------#


#Codificacion One-Hot para variables nominales
from sklearn.preprocessing import OneHotEncoder

#Por lo visto nom_5, ..., nom_9 son valores que tienen demasiada variabilidad
# y por las graficas estos estan distribuidos casi uniformemente por todo el datset lo que podria indicar
# que no son relevantes para el estudio de nuestro data set

# Seleccionar columnas nominales
columnas_nominales = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
# Aplicar One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
X_nominal_encoded = encoder.fit_transform(df[columnas_nominales])

# Crear un DataFrame con los nombres de las columnas codificadas
feature_names = encoder.get_feature_names_out(columnas_nominales)
X_nominal_encoded_df = pd.DataFrame(X_nominal_encoded, columns=feature_names)

print("\nCodificación One-Hot para variables nominales:")
print(X_nominal_encoded_df.head())

#-----------------------------------------------------------------------------#

#Codificacion para variables ordinales
from sklearn.preprocessing import OrdinalEncoder

# Seleccionar columnas ordinales
columnas_ordinales = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']  
# Definir el orden lógico de las categorías (si aplica)

ordinal_encoder = OrdinalEncoder()  # Sin especificar categorías
X_ordinal_encoded = ordinal_encoder.fit_transform(df[columnas_ordinales])

# Crear un DataFrame con las columnas codificadas
X_ordinal_encoded_df = pd.DataFrame(X_ordinal_encoded, columns=columnas_ordinales)

print("\nCodificación Ordinal para variables ordinales:")
print(X_ordinal_encoded_df.head())

#-----------------------------------------------------------------------------#
# Combinar todas las transformaciones
df_final = pd.concat([df[['id', 'bin_0', 'bin_1', 'bin_2']], X_label_endoded_df, X_nominal_encoded_df, X_ordinal_encoded_df, df[['day','month','target']] ], axis=1)

print("\nDataset final después de las transformaciones:")
print(df_final.head())

# Guardar el DataFrame final en un archivo CSV
csv_filename = 'train_encoded'
df_final.to_csv(csv_filename, index=False)
