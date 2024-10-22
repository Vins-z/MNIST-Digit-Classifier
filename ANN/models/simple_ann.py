from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def create_simple_ann():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
