
import numpy
import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(X_train,Y_train),(X_test, Y_test)=mnist.load_data()

rows, cols =28,28

X_train=X_train.reshape(X_train.shape[0],rows,cols,1)
X_test=X_test.reshape(X_test.shape[0],rows,cols,1)

input_shape=(rows,cols,1)

X_train=X_train.astype('float32')
X_train=X_train/255.0
X_test=X_test.astype('float32')
X_test=X_test/255.0

Y_train=tf.keras.utils.to_categorical(Y_train,10)
Y_test=tf.keras.utils.to_categorical(Y_test,10)

def build_lenet(input_shape):
    model=tf.keras.Sequential()
    #1_Convolution
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=input_shape))
    #1_Average_Pooling
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    #2_Convolution
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), activation='tanh'))
    #2_Average_Pooling
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    #1_flattening
    model.add(tf.keras.layers.Flatten())
    #1_Fully_Connected
    model.add(tf.keras.layers.Dense(units=120,activation='tanh'))
    #2_flattening
    model.add(tf.keras.layers.Flatten())
    #2_Fully_Connected
    model.add(tf.keras.layers.Dense(units=84,activation='tanh'))
    #Output
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

    #Compile_model
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(lr=0.1, decay=0.0), metrics=['accuracy'])
    return model

#Training
lenet=build_lenet(input_shape)
epochs=10
history=lenet.fit(X_train,Y_train,
                  epochs=epochs,
                  batch_size=128,
                  verbose=1)
loss,acc = lenet.evaluate(X_test, Y_test)
print('\033[92m''Accuracy: ', acc)

X_train=X_train.reshape(X_train.shape[0],28,28)
print('\033[95m''Train Data', X_train.shape,Y_train.shape)

X_test=X_test.reshape(X_test.shape[0],28,28)
print('\033[95m''Test Data', X_test.shape,Y_test.shape)

#Testing
image_index=12
plt.imshow(X_test[image_index].reshape(28,28),cmap='Greys')
pred=lenet.predict(X_test[image_index].reshape(1,rows,cols,1))
print('\033[93m'+"predicted Number:",pred.argmax())

