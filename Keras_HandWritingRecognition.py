from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from matplotlib import pyplot as plt
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint,History
import pickle


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert data into float32 as needed by tensorflow
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Convert the labels to categorical form
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


batch_size = 32
num_classes = 10
epochs = 1

hist = History()
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation='relu', padding = 'same', strides=(1,1), input_shape = ( 28, 28,1))) #why 1, 28, 28 doesnt work?
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

model.add(Conv2D(64, kernel_size = (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard_logs_callback = TensorBoard(log_dir = 'Graph', histogram_freq = 0, write_graph = True, write_images = True)
best_models_callback = ModelCheckpoint('./models/weights.hdf5', monitor = 'val_loss', save_best_only = True)

callbacks = []
callbacks.append(tensorboard_logs_callback)
callbacks.append(best_models_callback)

histor = model.fit(X_train, y_train, batch_size=128, epochs=2,validation_data=(X_test, y_test), callbacks=callbacks)
print(histor.history)

model.load_weights('./models/weights.02-0.03.hdf5')
histor = model.fit(X_train, y_train, batch_size=50, epochs=20000,validation_data=(X_test, y_test), callbacks=callbacks)

loss, accuracy = model.evaluate(X_test, y_test)


predictions = model.predict_classes(X_test, batch_size=50)
#print(predictions)

print(X_test)
for i in range(1, 10):
    im = X_test[i].reshape(28,28)
    plt.imshow(im)
    plt.show()
    c = y_test[i]
    print(c)
    print(predictions[i])

#with open('./models/hist.p', 'wb') as f:
 #   pickle.dump(histor.history, f)

#json_file = model.to_json()
#with open('model.json','w') as json_file:
 #   json_file.write(json_file)
#model.sample_weights('model.h5')

#model.load_weights()