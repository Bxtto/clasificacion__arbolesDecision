from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_final = pd.read_csv("train_encoded.csv")

X = df_final.drop(columns=['target'], axis=1)
y = df_final['target'].to_numpy()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Modelo: Árbol de decisión =====
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_train, y_train)
y_pred_arbol = arbol.predict(X_test)
acc_arbol = accuracy_score(y_test, y_pred_arbol)
print(f"Precisión Árbol de decisión: {acc_arbol:.2f}")

#Agregar aqui el codigo hecho en clase

scoring = {
    'accuracy':'accuracy',
    'precision_macro':'precision_macro',
    'recall_macro':'recall_macro',
    'f1_macro':'f1_macro'
}

results = cross_validate(arbol, X,y, cv=10,scoring=scoring)
print(f"Accuracy promedio: {results['test_accuracy'].mean():.4f}")
print(f"Precision promedio: {results['test_precision_macro'].mean():.4f}")
print(f"Recall promedio: {results['test_recall_macro'].mean():.4f}")
print(f"F1-score promedio: {results['test_f1_macro'].mean():.4f}")


# Visualizar el árbol
plt.figure(figsize=(10, 6))
plot_tree(arbol, filled=True, feature_names=X.columns, class_names=arbol.classes_.astype(str))
plt.title("Árbol de decisión - train")
plt.show()


# Probar diferentes criterios de división
criterios = ['gini', 'entropy']

for criterio in criterios:
    arbol = DecisionTreeClassifier(criterion=criterio, max_depth=3, random_state=42)
    arbol.fit(X_train, y_train)
    y_pred = arbol.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión Árbol de decisión ({criterio}): {acc:.4f}")


# ===== Modelo: K-Nearest Neighbors =====
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Precisión KNN: {acc_knn:.2f}")