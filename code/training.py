import sklearn
import pandas as pd
#Model_selection permite dividir el dataset en 2 subconjuntos: train y test
from sklearn.model_selection import train_test_split
#Librerias para el modelo de regresion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#Creacion de graficos
import matplotlib.pyplot as plt
from joblib import dump, load

#DATASET
dataframe = pd.read_csv("datos00.csv")
df_x = dataframe[['x']]
df_y = dataframe[['y']]

#GRAFICAR DATAFRAME
plt.plot(df_x,df_y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("REGRESION LINEAL")
plt.savefig("reg_lin.png")
plt.show()

#SEPARAR DATASET EN TRAIN(80%) Y TEST(20%)
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2, random_state=0)

#VERIFICAR EL PORCENTAJE A VERIFICAR
print ('Training: %d rows\nTest: %d rows' % (x_train.shape[0], x_test.shape[0]))

#ENTRENAR EL MODELO DE MACHINE LEARNING
model = LinearRegression()
model.fit(x_train, y_train)

#CALCULAR LA PRECISION
y_hat = model.predict(x_test)
acc = r2_score(y_test, y_hat)
print("Accuracy: %.2f" % acc)

#GUARDAR EL MODELO
dump(model, "model.joblib")