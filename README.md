# JSS_internship
# This project is for Japan Space System Internship 2020
# The project does not contain training dataset

To use this code:
 1. Install all required libraries in env.txt file.
 2. Put dataset in tho the following format.
 ```
 └── data
      ├── train
      │    ├── images
      │    └── masks
      ├── validation
      │    ├── images
      │    └── masks
      └── test
           ├── images
           └── masks
 ```
 3. Set training parameters in the bottom of main.py file.
 ```python
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
 ```
 - loss function can be set in the first line.
 - make sure image dimension is correct
 - the rest of parrameters can be modify to suit the training.
 \**all parameters under image size is just a dummy number pleast not use it for training
