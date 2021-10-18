import sklearn
from joblib import load

model = load("model.joblib")
while True:
    xs = []
    x = int(input("Valor de X: "))
    xs.append([x])
    prediction = model.predict(xs)
    print(prediction)