# Creación de un dataset artificial con categóricos
import pandas as pd
import numpy as np

np.random.seed(42)

# Definimos los valores posibles
colores = ['rojo', 'azul', 'verde']
formas = ['círculo', 'cuadrado', 'triángulo']
tamaños = ['pequeño', 'mediano', 'grande']
clases = ['A', 'B', 'C']

# Creamos el DataFrame
n = 10
df = pd.DataFrame({
    'color': np.random.choice(colores, n),
    'forma': np.random.choice(formas, n),
    'tamaño': np.random.choice(tamaños, n),
    'clase': np.random.choice(clases, n)
})

print(df)


# Codificación onehot

from sklearn.preprocessing import OneHotEncoder

X = df[['color', 'forma', 'tamaño']]

encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Intentamos obtener nombres de columnas de forma compatible
try:
    feature_names = encoder.get_feature_names_out(X.columns)
except AttributeError:
    feature_names = encoder.get_feature_names(X.columns)

X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

print("\n\n\n\nCodificación one hot")
print(X_encoded_df.head())



# Codificación ordinal

from sklearn.preprocessing import OrdinalEncoder

# Crear codificador OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Ajustar y transformar
X_ordinal = ordinal_encoder.fit_transform(X)

# Convertir a DataFrame para verlo mejor
X_ordinal_df = pd.DataFrame(X_ordinal, columns=X.columns)
print("\n\n\n\nCodificación ordinal")
print(X_ordinal_df)




