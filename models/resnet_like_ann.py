from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Add, BatchNormalization, Activation

def create_resnet_like_ann():
    def residual_block(x, units):
        y = Dense(units)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dense(units)(y)
        y = BatchNormalization()(y)
        out = Add()([x, y])
        out = Activation('relu')(out)
        return out

    inputs = Input(shape=(28, 28, 1))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
