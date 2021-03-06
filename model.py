from tensorflow import keras
from layers import *

def unet(pretrained_weights = None,input_size = (64,64,32,1),l_rate=0.01,loss_func=keras.losses.BinaryCrossentropy()):
    inputs = keras.layers.Input(input_size)
    conv1, pool1 = down_sampling(inputs, 8)
    conv2, pool2 = down_sampling(pool1, 16)
    conv3, pool3 = down_sampling(pool2, 32)
    conv4, pool4 = down_sampling(pool3, 64)
    conv5 = keras.layers.Conv2D(128, 3, padding = "same", activation = 'relu')(pool4)
    conv5 = keras.layers.Conv2D(128, 3, padding = "same", activation = 'relu')(conv5)
    up1 = up_sampling(conv5, conv4, 64)
    up2 = up_sampling(up1, conv3, 32)
    up3 = up_sampling(up2, conv2, 16)
    up4 = up_sampling(up3, conv1, 8)

    conv_last = keras.layers.Conv2D(2, 3, padding="same", activation="relu", kernel_initializer = 'he_normal')(up4)
    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(conv_last)

    model = keras.models.Model(inputs, outputs)

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=l_rate), loss = loss_func, metrics=['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
