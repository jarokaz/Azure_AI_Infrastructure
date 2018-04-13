
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization


def identity_block(X, f, filters, stage, block):
    """
    Identity block 
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides= (1,1), padding = 'same', name =conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X


def conv_block(X, f, filters, stage, block, s = 1):
    """
    Implementation of the convolutional block
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides= (1,1), padding = 'same', name =conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    
    return X

NUM_CLASSES=10
def model(img_input):
    
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    
    x = conv_block(x, 3, [64, 64, 64], stage=1, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 64], stage=1, block='b')
    x = identity_block(x, 3, [64, 64, 64], stage=1, block='c')
    x = identity_block(x, 3, [64, 64, 64], stage=1, block='d')
    x = identity_block(x, 3, [64, 64, 64], stage=1, block='e')
    x = identity_block(x, 3, [64, 64, 64], stage=1, block='f')

    x = conv_block(x, 3, [128, 128, 128], stage=1, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 128], stage=2, block='b')
    x = identity_block(x, 3, [128, 128, 128], stage=2, block='c')
    x = identity_block(x, 3, [128, 128, 128], stage=2, block='d')
    x = identity_block(x, 3, [128, 128, 128], stage=2, block='e')
    x = identity_block(x, 3, [128, 128, 128], stage=2, block='f')

    x = conv_block(x, 3, [256, 256, 256], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 256], stage=3, block='b')
    x = identity_block(x, 3, [256, 256, 256], stage=3, block='c')
    x = identity_block(x, 3, [256, 256, 256], stage=3, block='d')
    x = identity_block(x, 3, [256, 256, 256], stage=3, block='e')
    x = identity_block(x, 3, [256, 256, 256], stage=3, block='f')

    x = AveragePooling2D(pool_size=8, name='avg_pool')(x)

    x = Flatten()(x)
    logits = Dense(NUM_CLASSES,  name='fc10')(x)
    
    return logits
    
    
