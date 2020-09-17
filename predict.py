import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

## global parameters

def predict(MODEL, base_path, img_name, save_dir='./result'):
    image_path = os.path.join(base_path, 'images/', img_name)
    img = cv2.imread(image_path,1)

    mask_path = os.path.join(base_path, 'masks/', img_name)
    mask = cv2.imread(mask_path,0)

    pred = np.expand_dims(img, axis=0)
    prediction = MODEL.predict(pred)
    prediction = np.squeeze(prediction)

    if not os.path.exists('%s/'%save_dir):
        os.makedirs('%s/'%save_dir)
        os.makedirs('%s/grid/'%save_dir)
    save_grid(save_dir, img_name, img, mask, prediction)

def predict_folder(MODEL, base_path, save_dir='./result'):
    files = glob.glob('%simages/*'%base_path)
    for path in tqdm(files, desc='images in folder', leave=False):
        img_name = os.path.basename(path)
        predict(MODEL, base_path, img_name, save_dir)

def save_grid(save_dir, name, img, mask, pred):
    fig, a = plt.subplots()
    a.axis('off')
    fig.suptitle(name, fontsize = 16)

    a = fig.add_subplot(1, 3, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(img)
    a.set_title('Image', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(mask)
    a.set_title('Ground truth', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(pred)
    cv2.imwrite('%s/%s.png'%(save_dir, name), pred)
    a.set_title('Prediction', loc='center')
    a.axes.xaxis.set_visible(False)
    a.axes.yaxis.set_visible(False)

    plt.savefig('%s/grid/%s.png'%(save_dir, name))
    plt.close()