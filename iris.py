import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from img_datas import X_train, X_test, Y_train, Y_test


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

history = model.fit(X_train,Y_train, validation_data=(X_test, Y_test), epochs=500) #verbose=0



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy'])
plt.show()
