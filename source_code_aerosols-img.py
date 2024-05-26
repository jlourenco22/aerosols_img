# Bibliotecas Gerais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Carregar dados
train_data = pd.read_csv('/kaggle/input/aerosols-img/train.csv')
test_data = pd.read_csv('/kaggle/input/aerosols-img/test.csv')

# Remover colunas 'file_name_l1' e 'id'
train_data = train_data.drop(columns=['file_name_l1', 'id'])

#Correlação entre Features
train_data.corr() 

# Train e Validation Set
train_set, validation_set = train_test_split(train_data, test_size=0.3)


# Estatísticas descritivas
print("Estatísticas Descritivas do Dataset de Treino:")
print(train_set.describe())

# Estatísticas descritivas
print("Estatísticas Descritivas do Dataset de Validação:")
print(validation_set.describe())


# Verificar valores nulos
print("Verificar Missing Data:")
print(train_data.isnull().sum())



#Visualizar a distribuição normal das features
df_analise_dist = train_data.melt()

#Criar um FaceGrit com um histograma para cada feature do DataSet
g = sns.FacetGrid(df_analise_dist, col="variable", col_wrap=4, sharex=False, sharey=False, height=4)
g.map(sns.histplot, "value", kde=False, color='blue', bins=30)
plt.show()


#Visualizar a distribuição de outliers
df_analise_box_plot = train_data.melt()

#FaceGrit com os box Plot
g = sns.FacetGrid(df_analise_box_plot, col="variable", col_wrap=4, sharex=False, sharey=False, height=4)
g.map(sns.boxplot, "value")
plt.show()
        

# Visualizações (dependendo do tipo de dados, ajuste as visualizações)
sns.pairplot(train_data)
plt.show()
