import tensorflow as tf
import pandas as pd

iris_datas = "https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv"
iris = pd.read_csv(iris_datas)
iris = pd.get_dummies(iris)

inde_var = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]               #독립변수 
depe_var = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]          #종속변수

print(inde_var.shape,depe_var.shape)

model = 
X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)                         #softmax 분류할때 주로 사용하는것들
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy',
              metrics='accuracy')

model.summary()

model.fit(inde_var,depe_var, epochs=1000, verbose=0)
model.fit(inde_var,depe_var, epochs=10)

print(model.predict(inde_var))
print(depe_var)
