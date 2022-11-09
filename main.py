#Bibliotecas Python
import sys

import pandas as pd
import scipy
import numpy
import matplotlib
import pandas
import sklearn

#Carregar bibliotecas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Carregar através da URL
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names= attributes)
dataset.columns = attributes

#Dimensões do Dataset (shape)
print(dataset.shape)

#Analise os dados (head)
print(dataset.head(20))

#Resumo estatistico (descriptions)
print(dataset.describe())

#Distribuição de Classe (class distribution)
print(dataset.groupby('class').size())

#Visualização de Dados

## Graficos Univariados (box and whisker plots)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

## Graficos Univariados (histograms)
dataset.hist()
plt.show()

## Graficos Multivariados (scatter plot matrix)
scatter_matrix(dataset)
plt.show()

