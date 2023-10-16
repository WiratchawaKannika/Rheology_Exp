import PIL
import os
import glob
import shutil
import sys
import random
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_valid_train(fold0) :
    n_ran = 1
    empy_valid = pd.DataFrame(columns=list(fold0.columns))
    lst_cls = list(set(fold0['subclass']))
    for m in lst_cls:
        df_f =  fold0[fold0["subclass"]==m]
        lst_fold = list(set(df_f["fold"]))
        lst_fold.sort()
        for j in range(len(lst_fold)):
            DF_fold_i = df_f[df_f['fold']==lst_fold[j]]
            list_code =  list(set(DF_fold_i["Code"]))
            random.seed(0)
            lst_ran = random.sample(list_code, n_ran)
            for n in range(len(lst_ran)):
                AA = fold0[fold0["Code"]==lst_ran[n]]
                empy_valid = empy_valid.append(AA)
    ## Create DataFrame
    fold0.drop(index=empy_valid.index, inplace=True)
    fold0 = fold0.reset_index(drop=True)
    empy_valid = empy_valid.reset_index(drop=True)
    
    return fold0, empy_valid



### Get data Loder
train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    rotation_range=20,
                    brightness_range=[0.5, 1.5],
                    shear_range=0.5,
                    horizontal_flip = False,
                    fill_mode = 'nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


def Data_generator(IMAGE_SIZE, BATCH_SIZE, DFtrain, DFvalid, colums_y, class_mode_img):
    train_generator = train_datagen.flow_from_dataframe(
                    dataframe = DFtrain,
                    directory = None,
                    x_col = 'image_path',
                    y_col = colums_y,
                    target_size = (IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    color_mode= 'rgb',
                    class_mode =class_mode_img)

    test_generator = test_datagen.flow_from_dataframe(
                    dataframe = DFvalid,
                    directory = None,
                    x_col = 'image_path',
                    y_col = colums_y,
                    target_size = (IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    color_mode= 'rgb',
                    class_mode=class_mode_img)
    
    return train_generator, test_generator 
