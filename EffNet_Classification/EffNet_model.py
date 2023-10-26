import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from efficientnet.keras import EfficientNetB7 as Net
#load Check point
from tensorflow.keras.models import load_model



def build_modelB7(fine_tune, Numclasses):
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    """
    
    conv_base = Net(weights='imagenet')
    height = width = conv_base.input_shape[1]
    input_shape = (height, width, 3)
    #print(f"Input Shape: {input_shape}")
    
    # loading pretrained conv base model
    conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)
    # create new model with a new classification layer
    x = conv_base.output  
    global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
    dropout_layer_1 = layers.Dropout(0.50, name = 'head_dropout')(global_average_layer)
    prediction_layer = layers.Dense(Numclasses, activation='softmax',name = 'prediction_blood')(dropout_layer_1)
    ### lastlayer 
    model = models.Model(inputs= conv_base.input, outputs=prediction_layer, name = 'EffNetModel_Blood') 

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in conv_base.layers:
            layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return input_shape, model


def loadresumemodel(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    
    return input_shape, model



def model_block7Unfreze(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
    model.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name.startswith('block7'):
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    print('This is the number of trainable layers '
          'after freezing the block7 Layer:', len(model.trainable_weights))

    return input_shape, model



def model_block5Unfreze(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
    model.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block5a_se_excite':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    print('This is the number of trainable layers '
          'after freezing the block5a_se_excite Layer:', len(model.trainable_weights))

    return input_shape, model


