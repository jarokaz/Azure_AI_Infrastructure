from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers


def network_model(hidden_units, inputs, num_classes):
     
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_tensor=inputs)
                
    for layer in conv_base.layers:
        layer.trainable = False

    a = Flatten()(conv_base.output)
    a = Dense(hidden_units, activation='relu')(a)
    a = Dropout(0.5)(a)
    y = Dense(num_classes, activation='softmax')(a)
                                                           
    return y
