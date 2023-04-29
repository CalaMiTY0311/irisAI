import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris

iris_datas = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
#iris_datas = load_iris()

iris = pd.read_csv(iris_datas)
iris = pd.get_dummies(iris)

inde_var = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]               #독립변수 
depe_var = iris[['품종_setosa', '품종_versicolor', '품종_virginica']]          #종속변수

print(inde_var.shape,depe_var.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=[4]),
    tf.keras.layers.Dense(64, activation="swish"),
    tf.keras.layers.Dense(128, activation="swish"),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(
            loss='categorical_crossentropy',
            optimizer = 'Adam',
            metrics='accuracy'
            )

model.summary()

history = model.fit(np.array(inde_var),np.array(depe_var), epochs=500) #verbose=0

#print(model.predict(inde_var))
print(depe_var)

plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy'])
plt.show()
"""