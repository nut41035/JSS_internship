from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization

def down_sampling(Input, target_size, kernel_size = 3, padding = "same", activation = "relu", stride = 1):
    conv = Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation = activation, kernel_initializer = 'he_normal')(Input)
    conv = BatchNormalization()(conv)
    conv = Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation = activation, kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPool2D(pool_size=(2, 2))(conv)
    return conv, pool

def up_sampling(Input1, Input2, target_size, kernel_size = 3, padding = "same", activation = "relu", stride = 1):
    # Input1 is from previous layer
    # Input2 is layer to concat
    pool = UpSampling2D((2, 2))(Input1)
    merge = Concatenate()([pool,Input2])
    conv = Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation=activation, kernel_initializer = 'he_normal')(merge)
    conv = BatchNormalization()(conv)
    conv = Conv2D(target_size, kernel_size, padding = padding, strides = stride, activation=activation, kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    return conv
