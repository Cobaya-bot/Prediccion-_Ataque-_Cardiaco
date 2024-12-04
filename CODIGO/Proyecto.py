# DECLARAMOS LAS LIBRERIAS A UTILIZAR
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
# IGNORAMOS LAS ADVERTENCIAS
warnings.filterwarnings("ignore", category=DeprecationWarning)

# CARGAMOS LOS DATOS
df = pd.read_csv("D:/Prediccion-_Ataque-_Cardiaco/DATASET/heart.csv")
# MOSTRAMOS LAS 5 PRIMERAS COLUMNAS
print(df.head())

# ELIMINANDO DUPLICADOS
df.drop_duplicates(keep='first',inplace=True)

'''
#=================MOSTRANDO LOS DATOS EN GRAFICOS==========================================
# GRAFICO DE BARRAS SEGUN EL GENERO DE LA PERSONA
x=(df.sex.value_counts())
p = sns.countplot(data=df, x="sex", palette="muted")
plt.title("0: Femenino         1:Masculino")
plt.xlabel("Género")
plt.ylabel("Cantidad")
plt.show()


# GRAFICO DE BARRAS SEGUN EL TIPO DE DOLOR DE PECHO
x=(df.cp.value_counts())
p = sns.countplot(data=df, x="cp", palette="muted")
plt.title("0:Angina Típica  1:Angina Atípica  2:Dolor no Anginoso  3:Asintomático")
plt.xlabel("Tipo de Dolor de Pecho")
plt.ylabel("Cantidad")
plt.show()


# ANGINA INDUCIDA POR EJERCICIO
x=(df.exng.value_counts())
p = sns.countplot(data=df, x="exng", palette="muted")
plt.title("0:No           1:Si")
plt.xlabel("Angina Inducida por Ejercicio")
plt.ylabel("Cantidad")
plt.show()

# ELECTROCARDIOGRAMA
x=(df.restecg.value_counts())
p = sns.countplot(data=df, x="restecg", palette="muted")
plt.title("0:Normal 1:Presenta Anomalia 2:Hipertrofia Ventricular")
plt.xlabel("Electrocardiograma")
plt.ylabel("Cantidad")
plt.show()

# COLESTEROL
fig = plt.figure(figsize=(18,16))
gs = fig.add_gridspec(2,3)
ax = fig.add_subplot(gs[0,0])
ax.text(-0.25, 600, 'Colesterol(mg/dl)', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxenplot(ax=ax,y=df['chol'],palette=["#6aac90"],width=0.6)
ax.set_xlabel("")
ax.set_ylabel("")

# EDAD
plt.figure(figsize=(10,10))
sns.displot(df.age, color="red", label="EDAD", kde= True)
plt.xlabel("Edad")
plt.ylabel("Cantidad")
plt.legend()

# PRESION ARTERIAL
plt.figure(figsize=(20,20))
sns.displot(df.trtbps, label="Presión Arterial en Reposo", kde= True)
plt.xlabel("Presión Arterial")
plt.ylabel("Cantidad")
plt.legend()
#==============================================================================================
'''

# PROCESANDO DATOS
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# ENTRENAMIENTO Y PRUEBA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# ESCALANDO DATOS
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# UTILIZAMOS MODELO SVC
modelo = SVC()
modelo.fit(x_train, y_train)
predicted = modelo.predict(x_test)

# MOSTRAMOS LA PRECISIÓN
print("LA PRECISIÓN DE SVC ES : ", accuracy_score(y_test, predicted)*100, "%")
print("")

# CALCULAMOS LA MATRIZ DE CONFUSIÓN
conf = confusion_matrix(y_test, predicted)

# MOSTRAMOS LA MATRIZ DE CONFUSIÓN
#print("Matriz de Confusión:")
#print(conf)

# Visualizar la matriz de confusión usando un mapa de calor
plt.figure(figsize=(6, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('1:Más Posibilidades de Sufrir un Ataque Sardíaco')
plt.ylabel('0:Menos Posibilidades de Sufrir un Ataque Cardíaco')
plt.title('Matriz de Confusión')
plt.show()
