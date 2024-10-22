import struct
import numpy as np
import tensorflow as tf

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _ = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28, 1)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _ = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_data():
    train_images_path = 'ANN/MNIST-Image-Classifier/train-images.idx3-ubyte'
    train_labels_path = 'ANN/MNIST-Image-Classifier/train-labels.idx1-ubyte'
    test_images_path = 'ANN/MNIST-Image-Classifier/t10k-images.idx3-ubyte'
    test_labels_path = 'ANN/MNIST-Image-Classifier/t10k-labels.idx1-ubyte'

    x_train = load_mnist_images(train_images_path).astype('float32') / 255.0
    y_train = load_mnist_labels(train_labels_path)
    x_test = load_mnist_images(test_images_path).astype('float32') / 255.0
    y_test = load_mnist_labels(test_labels_path)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test
