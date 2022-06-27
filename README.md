- 👋 Hi, I’m @judicachu
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
judicachu/judicachu is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

Practica 2. Opcion 1 (Prediccion Precio de un Móvil)


Bob ha comenzado su propia empresa de telefonía móvil. Quiere dar una pelea dura a las grandes empresas como Apple, Samsung, etc.

No sabe cómo estimar el precio de los móviles que fabrica su empresa. En este competitivo mercado de telefonía móvil no se puede simplemente asumir cosas. Para resolver este problema, recopila datos de ventas de teléfonos móviles de varias empresas.

Bob quiere averiguar alguna relación entre las funciones de un teléfono móvil (p. ej., RAM, memoria interna, etc.) y su precio de venta. Pero no es tan bueno en Machine Learning. Así que necesita tu ayuda para resolver este problema.

En este problema, no tiene que predecir el precio real, sino un rango de precios que indica qué tan alto es el precio

Objetivo de la practica
Parte 1. En un Notebook construya un modelo para predecir el rango de precio de un móvil.
Parte 2. Publique y comparta su Notebook en Github.com
Parte 3. En base al trabajo desarrollado y los hallazasgos redacte, publique y comparta un articulo en medium.com
Criterios de Evaluación
Parte 1.

Seguir todos los pasos de un proceso ML:

Identificación del Problema
Importación de datos
EDA
Modelado (probar al menos tres algoritmos)
Evaluación (al menos dos métricas)
Hypertunning con Gridsearch
Selección del Modelo
Resultados y Conclusiones
Todo el código del proyecto esté contenido en un cuaderno o script de Jupyter o Google Colab.

Demuestre una ejecución y salida exitosas del código, sin errores.

Escriba código que esté bien documentado y use funciones y clases según sea necesario.

Parte 2.

El repositorio en Github debe conteener:

El cuaderno *.ipynb
README.md, que explique el problema, una descripcion del dataset, requerimientos y librerias para la correcta ejecución del codigo, Resultados y conclusiones
Licencia del codigo
Parte 3.

El articulo en medium debera contener las siguientes secciones:

Introduccion
Problema
Analisis Exploratorio explicando aspectos relevantes de lo encontrado
Modelado ML
Evaluación y Hipertunning
Resultados y Conclusiones
Incluir referencia on enlace al codigo disponible en su cuenta github.
Dataset
Puede descargar el dataset aqui: https://neuraldojo.org/media/mobile/archive.zip

Método de Entrega del trabajo:
Enviar e mi correo: necrus.aikon@gmail.com, Asunto: Practica 2 (Stroke) lo siguiente:

Enlace o archivo adjunto del codigo en colab
Enlace github
Enlace medium
Introducción
El problema de la predicción en cualquier campo es una tarea compleja. La recolección de información, en particular de tipo económicas ya sea de personas, empresas o países, se la realiza con fines de análisis para en el futuro llevar a cabo la planeación y la toma de decisiones.

Es lógico suponer que no se pueden tomar decisiones de políticas de un determinado proyecto, sin considerar la evaluación futura de todos aquellos elementos que lo condicionan. Por lo tanto la consideración del futuro en cualquier campo es ineludible para la toma de decisiones.

Problema
Predecir un rango de precios para determinar que tan alto es el precio de un teléfono móvil de acuerdo a funciones del mismo (RAM, memoria interna, etc).

Analisis Exploratorio
Este análisis explicando aspectos relevantes de lo encontrado.

Importación de las librerias
[2]
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

[3]
import pandas as pd
import numpy as np

Importamos los Datos
[8]
df_train =  pd.read_csv('train.csv')
Análisis Exploratorio (EDA) Simple
[12]
0 s
df_train.head(20)

[14]
0 s
df_train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   int64  
 1   blue           2000 non-null   int64  
 2   clock_speed    2000 non-null   float64
 3   dual_sim       2000 non-null   int64  
 4   fc             2000 non-null   int64  
 5   four_g         2000 non-null   int64  
 6   int_memory     2000 non-null   int64  
 7   m_dep          2000 non-null   float64
 8   mobile_wt      2000 non-null   int64  
 9   n_cores        2000 non-null   int64  
 10  pc             2000 non-null   int64  
 11  px_height      2000 non-null   int64  
 12  px_width       2000 non-null   int64  
 13  ram            2000 non-null   int64  
 14  sc_h           2000 non-null   int64  
 15  sc_w           2000 non-null   int64  
 16  talk_time      2000 non-null   int64  
 17  three_g        2000 non-null   int64  
 18  touch_screen   2000 non-null   int64  
 19  wifi           2000 non-null   int64  
 20  price_range    2000 non-null   int64  
dtypes: float64(2), int64(19)
memory usage: 328.2 KB
[16]
0 s
df_train.shape
(2000, 21)
[17]
0 s
df_train.head()

[ ]
df_test.shape
(1000, 21)
[ ]
df_train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   battery_power  2000 non-null   int64  
 1   blue           2000 non-null   int64  
 2   clock_speed    2000 non-null   float64
 3   dual_sim       2000 non-null   int64  
 4   fc             2000 non-null   int64  
 5   four_g         2000 non-null   int64  
 6   int_memory     2000 non-null   int64  
 7   m_dep          2000 non-null   float64
 8   mobile_wt      2000 non-null   int64  
 9   n_cores        2000 non-null   int64  
 10  pc             2000 non-null   int64  
 11  px_height      2000 non-null   int64  
 12  px_width       2000 non-null   int64  
 13  ram            2000 non-null   int64  
 14  sc_h           2000 non-null   int64  
 15  sc_w           2000 non-null   int64  
 16  talk_time      2000 non-null   int64  
 17  three_g        2000 non-null   int64  
 18  touch_screen   2000 non-null   int64  
 19  wifi           2000 non-null   int64  
 20  price_range    2000 non-null   int64  
dtypes: float64(2), int64(19)
memory usage: 328.2 KB
[18]
0 s
df_train.describe().T

[19]
0 s
#Verificamos si tenemos datos nulos 
df_train.isnull().sum()
battery_power    0
blue             0
clock_speed      0
dual_sim         0
fc               0
four_g           0
int_memory       0
m_dep            0
mobile_wt        0
n_cores          0
pc               0
px_height        0
px_width         0
ram              0
sc_h             0
sc_w             0
talk_time        0
three_g          0
touch_screen     0
wifi             0
price_range      0
dtype: int64
[ ]
df_train.isnull().sum()
id               0
battery_power    0
blue             0
clock_speed      0
dual_sim         0
fc               0
four_g           0
int_memory       0
m_dep            0
mobile_wt        0
n_cores          0
pc               0
px_height        0
px_width         0
ram              0
sc_h             0
sc_w             0
talk_time        0
three_g          0
touch_screen     0
wifi             0
dtype: int64
[21]
0 s
df_train.columns
Index(['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'],
      dtype='object')
[22]
0 s
sns.distplot(df_train['dual_sim'])

Division datos para entrenamiento y test
[23]
0 s
from sklearn.model_selection import train_test_split 
[24]
#Division de los datos
train, test = train_test_split(df_train, test_size = 0.30)

print("Ejemplos usados para entrenar: ", len(train))
print("Ejemplos usados para test: ", len(test))
Ejemplos usados para entrenar:  1400
Ejemplos usados para test:  600
[ ]
##Veamos distribuciones de los datos
dist_workclass = df_test['battery_power'].value_counts()
sns.barplot(x= dist_workclass, y = dist_workclass.index)

[ ]
train_.head()

[ ]
test_.head()

[ ]
sns.pairplot(data = train, diag_kind="battery_power")

[27]
X = train.iloc[:,1:]
y = train['battery_power']
X_test = test.iloc[:,1:]
y_test = test['ram']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, test_size = 0.3)
[28]
0 s
#Verificamos la distribución de nuestra variable objetivo
test['touch_screen'].value_counts()
1    304
0    296
Name: touch_screen, dtype: int64
[39]
0 s
# División de los datos en train y test
# ==============================================================================
X = train[['dual_sim']]
y = train['battery_power']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

Modelado ML
3 algoritmos

[40]
0 s
#1er algoritmo Regresión Lineal
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

LinearRegression()
[41]
0 s
# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))
Intercept: [1254.34115523]
Coeficiente: [('dual_sim', -22.623840747024527)]
Coeficiente de determinación R^2: 0.0005513679321506038
/usr/local/lib/python3.7/dist-packages/sklearn/base.py:444: UserWarning: X has feature names, but LinearRegression was fitted without feature names
  f"X has feature names, but {self.__class__.__name__} was fitted without"
Prueba en data test

[42]
1 s
# Error de test del modelo 
# ==============================================================================
# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")
/usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
  import pandas.util.testing as tm
[[1254.34115523]
 [1231.71731449]
 [1231.71731449]]

El error (rmse) de test es: 448.98700181818305
[49]
0 s
#2do algoritmo KNN
punto_nuevo = {'touch_screen': [1],
               'battery_power': [1080]}
              
punto_nuevo = pd.DataFrame(punto_nuevo)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
X = df_train[['touch_screen', 'battery_power']]
y = df_train[['ram']]
knn.fit(X, y)
prediccion = knn.predict(punto_nuevo)
print(prediccion)
[325]
/usr/local/lib/python3.7/dist-packages/sklearn/neighbors/_classification.py:198: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  return self._fit(X, y)
[ ]
print("Coef:",reg.coef_)
print("Intercept:",reg.intercept_)
Coef: [-1.08209190e+01  1.13657913e+01 -8.36264151e-01  1.23045466e+00
 -1.67970079e-01 -1.93733249e-01 -6.09899721e-01 -2.29392959e+00
  7.33593623e+00  1.34167479e+01  6.44787504e+02]
Intercept: 1929.38364200032
[53]
0 s
#3 algoritmo árbol de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)


DecisionTreeClassifier()
[55]
0 s
print(
  " clasificados al separar de dual sim si(1) y no(0)",
  train.loc[(train['dual_sim']==1) & (train['ram']>100),:].shape[0], "\n",
  "clasificados al separar :",
  train.loc[(train['dual_sim']==0) & (train['ram']>1000),:].shape[0]
)
 clasificados al separar de dual sim si(1) y no(0) 714 
 clasificados al separar : 547
[54]
1 s
# Super Vector Machine
from sklearn.svm import SVC
clf_svc = SVC(gamma = 'auto', kernel = 'sigmoid')
clf_svc.fit(X_train,y_train)
/usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
SVC(gamma='auto', kernel='sigmoid')
[ ]

# Imprimimos Scores
print("Train Score:", clf_svc.score(X_train,y_train))
print("Validation Score:", clf_svc.score(X_valid,y_valid))
Train Score: 0.0035714285714285713
Validation Score: 0.0
[ ]
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(X_train,y_train)
KNeighborsClassifier()
[ ]
# Imprimimos Scores
print("Train Score:", clf_knn.score(X_train,y_train))
print("Validation Score:", clf_knn.score(X_valid,y_valid))
Train Score: 0.2042857142857143
Validation Score: 0.0016666666666666668
Seleccionamos el Modelo
[ ]
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

clf_mnb = MultinomialNB()
clf_mnb.fit(X_train,y_train)

clf_bnb = BernoulliNB()
clf_bnb.fit(X_train,y_train)
BernoulliNB()
Evaluación
2 métricas

[ ]
# Imprimimos Scores
print("Multinomial Train Score:", clf_mnb.score(X_train,y_train))
print("Multinomial Validation Score:", clf_mnb.score(X_valid,y_valid))
Multinomial Train Score: 0.32785714285714285
Multinomial Validation Score: 0.0016666666666666668
[ ]
#Revisemos algunos valores maximos y minimos para entender las escalas de nuestros atributos
X.describe().T[['min','max']]

[ ]
data_pred = reg.predict(X)
print(data_pred)
[ 879.73882697 1042.11410575 1064.24953235 ... 1522.63702167 1210.65022909
 1235.09919615]
# Resultados y Conclusiones
Seleccionado el modelo para la predicción, tenemos el resultado de la predicción solicitada.

Se puede ver que varios algoritmos se pueden aplicar a este caso, sin embargo de acuerdo a lo requerido se toma la decisión de cual se escoge para el caso.
