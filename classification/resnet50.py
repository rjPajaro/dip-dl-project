from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense

def resnet_block(inputs, filters, kernel_size, strides=(1, 1), activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = inputs
    if strides != (1, 1) or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x

def getResNet50(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = resnet_block(x, 64, (3, 3))
    x = resnet_block(x, 64, (3, 3))
    x = resnet_block(x, 64, (3, 3))
    
    x = resnet_block(x, 128, (3, 3), strides=(2, 2))
    x = resnet_block(x, 128, (3, 3))
    x = resnet_block(x, 128, (3, 3))
    x = resnet_block(x, 128, (3, 3))
    
    x = resnet_block(x, 256, (3, 3), strides=(2, 2))
    x = resnet_block(x, 256, (3, 3))
    x = resnet_block(x, 256, (3, 3))
    x = resnet_block(x, 256, (3, 3))
    x = resnet_block(x, 256, (3, 3))
    x = resnet_block(x, 256, (3, 3))
    
    x = resnet_block(x, 512, (3, 3), strides=(2, 2))
    x = resnet_block(x, 512, (3, 3))
    x = resnet_block(x, 512, (3, 3))
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)  # Binary classification with sigmoid activation
    
    model = Model(inputs, x)
    return model