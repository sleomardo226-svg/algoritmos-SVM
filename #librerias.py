#librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons
from mpl_toolkits.mplot3d import Axes3D


#generar datos
X, y = make_moons(n_samples=200, centers=2, random_state=0.1, cluster_std=42)

#mostrar datos en un dataframe
data = pd.DataFrame(X, columns=['x1', 'x2'])
data['clase'] = y
print(data.head())

#entrenamiento del modelo SVM 3D

#crear el modelo svm no lineal
modelo = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

#entrenamiento al modelo
modelo.fit(X, y)

#grafica

plt.figure(figsize=(8, 6))
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=60, edgecolors='k')
plt.title('clasificación SVM no lineal')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

#crear un grid

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100))

#calcular valores de decisiones 
Z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#grafica en 3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

#superficie del modelo 
ax.plot_surface(xx, yy, Z, cmap='coolwarm', alpha=0.6)

#puntos de datos
ax.scatter(X[:, 0], X[:, 1], modelo.decision_function(X),
           c=y, cmap='coolwarm', edgecolor='k', s=60)

ax.set_title('SVM no lineal en 3D')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Función de decisión(Z)')
plt.show()
