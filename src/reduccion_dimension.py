import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from prince import PCA
from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

datos = pd.read_csv('data/gimnasio.csv', delimiter = ';', decimal = ".")
pred  = datos["Experiencia"]
datos = datos.drop(['Experiencia'], axis=1)
datos = pd.get_dummies(datos)
datos.dtypes

escalar = StandardScaler()
datos_escalados = escalar.fit_transform(datos)
datos_escalados = pd.DataFrame(datos_escalados)
datos_escalados.columns = datos.columns
datos_escalados.index = datos.index
datos_escalados

pca = PCA(n_components = 5)
pca.fit(datos_escalados)
individuos = pca.row_coordinates(datos_escalados)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

umap = UMAP(n_components = 2, n_neighbors = 325)
individuos = umap.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='random')
individuos = tsne.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()
