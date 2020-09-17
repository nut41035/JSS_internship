import tensorflow as tf
import os
import datetime

from model import *
from data import *
from predict import *

from tensorflow.compat.v1 import ConfigProto, GPUOptions, InteractiveSession

def train_and_eval(params):
    # session setting
    """ os.environ['TF_CPP_MIN_LOG_LEVEL']
      0 = all messages are logged (default behavior)
      1 = INFO messages are not printed
      2 = INFO and WARNING messages are not printed
      3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    ## program parameter
    BASE_DIR = params[0]
    TRAIN_DIR_PATH = BASE_DIR + 'train/'
    VALIDATION_DIR_PATH = BASE_DIR + 'validation/'
    seed = params[1]
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = params[2] + time_stamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    file_writer = tf.summary.create_file_writer(log_dir)

    ## training parameter
    loss_func = params[3]
    input_size = params[4]
    steps_per_epoch = params[5]
    EPOCHS = params[6]
    BS = params[7] 
    IMAGE_COUNT = params[8] 
    VALIDATION_COUNT = params[9]
    learning_rate = params[10]

    ## construct training and validation set
    training_data = DataGenerator(TRAIN_DIR_PATH, batch_size=BS, image_size=input_size[0])
    validating_data = DataGenerator(VALIDATION_DIR_PATH, batch_size=BS, image_size=input_size[0])

    ## load model
    model = unet(input_size = input_size,loss_func=loss_func,l_rate=learning_rate)
    model.summary()
    print('#### Model loaded')

    ## training begin
    model.fit_generator(training_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS,
                    validation_data=validating_data,
                    callbacks=[tensorboard_callback])

    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    model.save("model/UNet_%s.h5" %time_stamp)
    print("model saved at   model/UNet_%s.h5"%time_stamp)

    text = 'UNet_%s.h5\n\
            Learning rate: %s\n\
            Image size: %s\n\
            Epoch: %s\n\
            Batch size: %s\n\
            Step per epoch: %s\n'\
            %(time_stamp, learning_rate, input_size, steps_per_epoch, BS, EPOCHS)
    with open("./log.txt", "a") as myfile:
        myfile.write(text)
    file_writer.close()
    InteractiveSession.close(session)

    ## prediction begin
    predict_folder(model, '%stest/'%BASE_DIR, save_dir='./result/%s'%(time_stamp))
    InteractiveSession.close(session)

if __name__ == "__main__":
    loss = tf.keras.losses.BinaryCrossentropy()
    params = [
        './data/',    #Base directory
        1,            #Seed (daefault 1)
        './logs/',    #Log directory
        loss,
        (256,256,3),  #Input image dimension
        1,            #Step per epoch
        20,            #Number of epoch to train
        1,            #Batch size
        4,            #Total number of training images
        2,            #Total number of validation images
        0.05,         #Learning rate 
    ]
    train_and_eval(params)
