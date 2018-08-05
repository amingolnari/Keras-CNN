import warnings
warnings.filterwarnings("ignore",category = FutureWarning)
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD

def reset_graph(seed = 42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def export_model(MODEL_NAME, input_node_name, output_node_name):

    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')
    
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print(' ')
    print('Graph has been saved')
    print(' ')


def main():
    path = 'E:/Face Data/FaceData/'
    Width = 100
    Height = 100
    image_channel = 1

    if (image_channel == 1):
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'
      
    ## "Read from directory"
    gen = image.ImageDataGenerator()
    ## "Train images"
    train_batch = gen.flow_from_directory(path + 'Train', target_size = (Width, Height), shuffle = False, color_mode = color_mode,
        batch_size = 600, class_mode = 'categorical')

    ## "Test images"
    test_batch = gen.flow_from_directory(path + 'Test', target_size = (Width, Height), shuffle = False, color_mode = color_mode,
        batch_size = 300, class_mode = 'categorical')

    ## "Validation images"
    valid_batch = gen.flow_from_directory(path + 'Validation', target_size = (Width, Height), shuffle = True, color_mode = color_mode,
        batch_size = 300, class_mode = 'categorical')

    ## "Load data and labels"
    X_train, Y_train = next(train_batch)
    X_test, Y_test = next(test_batch)
    X_val, Y_val = next(valid_batch)

    ## "Reshape and normalize"
    X_train = X_train.reshape(600, Width , Height , image_channel).astype('float32')
    X_test = X_test.reshape(300, Width , Height , image_channel).astype('float32')
    X_val = X_test.reshape(300, Width , Height , image_channel).astype('float32')

    X_train /= 255
    X_test /= 255
    X_val /= 255

    reset_graph()

    ## "Creat model"
    MODEL_NAME = 'FaceFaculty'
    model = Sequential(name = MODEL_NAME)

    ## "First Layer"
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu',
            name = 'Input', padding = 'same',
            input_shape = (Width, Height, image_channel))) # Input Layer
    model.add(Conv2D(filters = 128, kernel_size = (5, 5), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (4, 4), padding = 'same'))
    model.add(BatchNormalization(trainable=False))
    
    ## "Second Layer"
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (3, 3), padding = 'same'))
    model.add(BatchNormalization(trainable=False))

    ## "Fully Connected"
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation = 'tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(100, activation = 'softmax', name = 'endNode'))

    ## "Show model summary"
    model.summary()

    ## "Compile options"
    model.compile(loss = 'categorical_crossentropy', 
       optimizer = SGD(lr = 0.01, momentum = 0.9), metrics = ['accuracy'])
    
    ## "Train"
    batch_size = 10
    epoch = 1

    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epoch, 
             verbose = 1,validation_data = (X_val, Y_val))

    ## "Test"
    score, acc = model.evaluate(X_test, Y_test, batch_size = batch_size)

    print('Test accuracy : ', acc)
    ## "Save model as h5 file"
    model.save('model.h5')
    ## "Load model from h5 file"
    newmodel = load_model('model.h5')
    ## "Export model to protobuffer file"
    export_model(MODEL_NAME, "Input_input", "endNode/Softmax")

if __name__ == "__main__":
    main()