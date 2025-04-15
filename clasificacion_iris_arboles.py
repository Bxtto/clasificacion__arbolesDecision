from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Modelo: Árbol de decisión =====
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_train, y_train)
y_pred_arbol = arbol.predict(X_test)
acc_arbol = accuracy_score(y_test, y_pred_arbol)
print(f"Precisión Árbol de decisión: {acc_arbol:.2f}")

#Agregar aqui el codigo hecho en clase

# Visualizar el árbol
plt.figure(figsize=(10, 6))
plot_tree(arbol, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Árbol de decisión - Iris")
plt.show()

# ===== Modelo: K-Nearest Neighbors =====
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Precisión KNN: {acc_knn:.2f}")


