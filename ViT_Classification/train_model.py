import os
import pandas as pd
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras import layers
from keras import models
from DataLoader import split_valid_train, Data_generator
from ViTModel import build_ViTModel, loadresumemodel
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import imageio
from keras.optimizers import Adam
import argparse
#load Check point
from tensorflow.keras.models import load_model



def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    my_parser.add_argument('--numclass', type=int, default=1, help='[1, 3]')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/code/Rheology2023/Rheology_Blood/Dataset_Rheology_Blood_Viscosity_HN_NBL-2dFFTdataset-3channels-6Fold-splitclass.csv')
    my_parser.add_argument('--save_dir', type=str, help='Main Output Path', default="/media/tohn/HDD/rheology2023/ViTModel/Classification/Blood_Viscosity")
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--fold', type=int, help='[For 1 Binary classes => fold 1-6, and 3 classes => fold 1-3]')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    my_parser.add_argument('--size', type=int, default=384, help='[224, 384]')
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
    numclass = args.numclass
    ## train seting
    num_epochs = args.epochs
    IMAGE_SIZE = args.size
    learing_rate = args.lr
    
    ## import dataset
    df_2dFFT = pd.read_csv(data_path)
    print(f"Dataset set: {df_2dFFT.shape[0]} 2dFFT images")
    DFtrain = df_2dFFT[df_2dFFT['fold']!=fold].reset_index(drop=True)
    DFvalid = df_2dFFT[df_2dFFT['fold']==fold].reset_index(drop=True)
    ## Split train, validation set
    #DFtrain, DFvalid = split_valid_train(train_2dFFT)
    print(f"[INFO]: For Train Set : With Shape {DFtrain.shape}")
    print(f"[INFO]: For Validation Set : With Shape {DFvalid.shape}")
    ### Get data Loader
    if numclass == 3:
        colums_y = "subclass"
        class_mode_img = 'categorical'
        loss_smooth = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2)
        metrics_acc = 'accuracy'
    elif numclass == 1:
        colums_y = "classes_binary"
        class_mode_img = 'raw'
        loss_smooth = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.2)
        metrics_acc = 'binary_accuracy'
        
    train_generator, test_generator = Data_generator(IMAGE_SIZE, BATCH_SIZE, DFtrain, DFvalid, colums_y, class_mode_img)
    
    ### Create Model 
    if args.resume :
        input_shape, vit_model = loadresumemodel(args.checkpoint_dir)
    else:    
        vit_model = build_ViTModel(fine_tune=False, IMAGE_SIZE=IMAGE_SIZE, Numclass=numclass)
    vit_model.summary()
    print('='*100)
    
    ## Set mkdir TensorBoard 
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    ##*** mkdir Modelname 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    modelName = f"ViTb32_{args.numclass}Class_{_fold}_{_R}.h5"
    Model2save = f'{modelNamemkdir}/{modelName}'
    ##*** on_epoch_end
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                self.model.save(f'{root_Metrics}{modelName}')
                return
    # For tracking Quadratic Weighted Kappa score and saving best weights
    metrics = Metrics()
    
    #Compile model
    from keras.optimizers import Adam
    vit_model.compile(loss=loss_smooth,
                       optimizer=Adam(learing_rate, decay=learing_rate),
                       metrics=[metrics_acc])
    
    #Training 
    history_freeze = vit_model.fit(train_generator,
                       epochs = num_epochs, 
                       validation_data = test_generator,
                       callbacks = [metrics, tensorboard_cb]
                       )
    
    
    # Save model as .h5        
    vit_model.save(Model2save)
    ### print
    print(f"Save EfficientNet model {args.numclass} Classification : {Model2save}")
    print(f"*"*100)
    
    
## Run Function 
if __name__ == '__main__':
    main()
    
    
    
        
        

    
    
    
    