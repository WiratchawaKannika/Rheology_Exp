import tensorflow as tf
#import tensorflow_addons as tfa
import glob, warnings
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import callbacks as callbacks_
from tensorflow.keras import layers
from keras import models
from vit_keras import vit, utils
from tensorflow.keras.models import load_model


#####################################
### ViT model with the functional ###
#####################################

def build_ViTModel(fine_tune, IMAGE_SIZE, Numclass):
    if Numclass == 1:
        activation_fun = 'sigmoid'
    elif Numclass == 3:
        activation_fun = 'softmax'
    
    
    """
    :param fine_tune (bool): Whether to train the hidden layers or not. 
                   Numclass: Number of Class to train ; if Numclass=1 train binary classification (class: HN and NBL)
                                                        if Numclass=3 train 3 classification (class: No_Splenectomy, Splenectomy and Normal)
    """
    
    vit_model = vit.vit_b32(image_size=IMAGE_SIZE, classes=Numclass, activation=activation_fun, pretrained = True, 
                            include_top = False, pretrained_top = False)
    print('[INFO]: Loading pre-trained weights')
    x = vit_model.get_layer('ExtractToken').output
    ### add the tail layer ###  
    Flatten_layer1 = layers.Flatten()(x)
    BatchNormalization_layer1 = layers.BatchNormalization(name='BatchNormalization_1')(Flatten_layer1)
    Dense_layer = layers.Dense(64, activation='gelu',name='Dense_1')(BatchNormalization_layer1)
    Dense_layer1 = layers.Dense(32, activation='gelu',name='Dense_2')(Dense_layer)
    BatchNormalization_layer2 = layers.BatchNormalization(name='BatchNormalization_2')(Dense_layer1)
    Dense_layer2 = layers.Dense(Numclass, activation=activation_fun, name='Pred_blood')(BatchNormalization_layer2)
        
    model = models.Model(inputs= vit_model.input, outputs=[Dense_layer2], name = 'ViT_BloodClass') 
    
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in vit_model.layers:
            layer.trainable = False
        print('This is the number of trainable layers '
                  'after freezing the conv base:', len(model.trainable_weights))
    print('-'*100)

    return model
    
    

def loadresumemodel(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    
    return input_shape, model


