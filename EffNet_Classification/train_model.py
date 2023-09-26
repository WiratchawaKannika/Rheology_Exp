import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from EffNet_model import build_modelB7, loadresumemodel, model_block7Unfreze, model_block5Unfreze
from DataLoader import Data_generator, split_valid_train
#load Check point
from tensorflow.keras.models import load_model
import argparse


def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=0, help='Number GPU 0,1')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/codes_AI/Rheology_Blood/Dataset_Rheology_Blood_Viscosity_HN_NBL-2dFFTdataset-3channels-6Fold.csv')
    my_parser.add_argument('--save_dir', type=str, help='Main Output Path', default="/media/SSD/rheology2023/EffNetB7Model/Classification/Blood_Viscosity")
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--fold', type=int, help='[fold 1-10]')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
    my_parser.add_argument('--tensorName', type=str ,default="TensorBoard")
    my_parser.add_argument('--epochendName', type=str ,default="on_epoch_end")
    my_parser.add_argument('--FmodelsName', type=str ,default="models")
    
    args = my_parser.parse_args()
    
    ## set gpu
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## get my_parser
    save_dir = args.save_dir
    name = args.name
    R = args.R
    _R = f'R{R}'
    fold = args.fold
    _fold = f"fold{fold}"
    ### create rootbase 
    root_base = f'{save_dir}/{name}/{_fold}/{_R}'
    os.makedirs(root_base, exist_ok=True)
    data_path = args.data_path
    BATCH_SIZE = args.batchsize
    ## train seting
    lr = args.lr
    num_epochs = args.epochs
    
    ### Create Model 
    if args.resume :
         input_shape, model = loadresumemodel(args.checkpoint_dir)
    elif args.R == 2:
        input_shape, model = model_block5Unfreze(args.checkpoint_dir)
    elif args.resume and args.R == 2:
        input_shape, model = loadresumemodel(args.checkpoint_dir)
    else:    
        input_shape, model = build_modelB7(fine_tune=True)
    ##get images size 
    height = width = input_shape[1]
    IMAGE_SIZE = (height, width)
    ### model summary
    model.summary()
    print('='*100)
    
    ## import dataset
    df_2dFFT = pd.read_csv(data_path)
    train_2dFFT = df_2dFFT[df_2dFFT['fold']!=fold].reset_index(drop=True)
    print(f"Dataset set: {train_2dFFT.shape[0]} 2dFFT images")
    ## Split train, validation set
    DFtrain, DFvalid = split_valid_train(train_2dFFT)
    print(f"[INFO]: For Train Set : With Shape {DFtrain.shape}")
    print(f"[INFO]: For Valid Set : With Shape {DFvalid.shape}")
    ### Get data Loder
    train_generator, test_generator = Data_generator(IMAGE_SIZE, BATCH_SIZE, DFtrain, DFvalid)
    
    ## Set mkdir TensorBoard 
    ##root_logdir = f'/media/SSD/rheology2023/VitModel/Regression/tensorflow/ExpTest/R1/Mylogs_tensor/'
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    ##*** mkdir Modelname 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    modelName = f"EffNetB7_HN-NBL_Class_{_fold}_{_R}.h5"
    Model2save = f'{modelNamemkdir}/{modelName}'
    ##*** 
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                self.model.save(f'{root_Metrics}{modelName}')
                return
    
    # For tracking Quadratic Weighted Kappa score and saving best weights
    metrics = Metrics()
    
    #Training
    from keras.optimizers import Adam
    
    lr=1e-4
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    
    ## Fit model 
    model.fit(train_generator, epochs = num_epochs, 
                validation_data = test_generator,
                 callbacks = [metrics, tensorboard_cb])
    
    
    # Save model as .h5        
    model.save(Model2save)
    ### print
    print(f"Save EfficientNet model Binary Classification : {Model2save}")
    print(f"*"*100)
    
## Run Function 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    





