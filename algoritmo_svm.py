#librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


#generar datos
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

#mostrar datos en un dataframe
data = pd.DataFrame(X, columns=['x1', 'x2'])
data['clase'] = y
print(data.head())

#entrenamiento del modelo SVM

#crear el modelo svm lineal
modelo = svm.SVC(kernel='linear', C=1.0) 

#entrenamiento al modelo
modelo.fit(X, y)

#obtener los coeficientes del plano 
w = modelo.coef_[0]
b = modelo.intercept_[0]

#calcular la frontera de decision 
x_linea = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
y_linea = -(w[0]/w[1])*x_linea - b/w[1]

#calcular margenes usando las líneas de decisión w·x + b = ±1
y_margen_sup = -(w[0]/w[1])*x_linea - (b - 1)/w[1]
y_margen_inf = -(w[0]/w[1])*x_linea - (b + 1)/w[1]

#grafica

plt.figure(figsize=(8, 6))
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=60, edgecolors='k')
plt.plot(x_linea, y_linea, 'k-', label='Frontera de decisión')
plt.plot(x_linea, y_margen_sup, 'k--')
plt.plot(x_linea, y_margen_inf, 'k--')
plt.scatter(modelo.support_vectors_[:, 0], modelo.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='k', label='vectores de soporte')
plt.title('clasificación SVM con frontera y márgenes')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
