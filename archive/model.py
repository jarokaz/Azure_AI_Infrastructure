'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input


def network_model(input_shape, input_name, num_classes, optimizer):
  
  inputs = Input(shape=input_shape, name=input_name)
  x = Conv2D(32, (3, 3), padding='same')(inputs)
  x = Activation('relu')(x)
  x = Conv2D(32, (3, 3))(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x)

  x = Conv2D(64, (3, 3), padding='same')(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3))(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x)

  x = Flatten()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(num_classes)(x)
  y = Activation('softmax')(x)
  
  model = Model(inputs=inputs, outputs=y)
  
  loss = 'categorical_crossentropy'
  metrics = ['accuracy']
  
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  
  return model
