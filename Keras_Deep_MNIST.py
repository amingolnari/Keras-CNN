import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.datasets import mnist

def LoadData():
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1).astype('float32')
    Xtrain = Xtrain / 255
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1).astype('float32')
    Xtest = Xtest / 255
    Ytrain = tf.keras.utils.to_categorical(Ytrain, 10)
    Ytest = tf.keras.utils.to_categorical(Ytest, 10)

    return (Xtrain, Xtest), (Ytrain, Ytest)

def BuildModel():
    model = Sequential()
    model.add(Conv2D(filters = 128, kernel_size = (5, 5),
                     activation = 'relu',
                     padding = 'same',
                     input_shape = (28, 28, 1)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                     padding = 'same',
                     activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                     activation = 'relu',
                     padding = 'same'))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(300, activation = 'tanh'))
    model.add(Dense(200, activation = 'sigmoid'))
    model.add(Dense(10, activation = 'softmax'))

    return model

def main():
    # Load dataset image and Labels
    (Xtrain, Xtest), (Ytrain, Ytest) = LoadData()
    
    # Build Model
    model = BuildModel()
    model.summary()

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = SGD(lr = 0.1),
                  metrics = ['accuracy'])
    # Train Model
    History = model.fit(Xtrain, Ytrain,
                        batch_size = 200,
                        epochs = 1,
                        validation_data = (Xtest, Ytest))
    # Test Model
    score, acc = model.evaluate(Xtest, Ytest)
    print('Test Accuracy : ', acc)

    return

if __name__ == "__main__":
    main()
