'''A simple deep CNN on that gets around 75% validation accuracy on CIFAR-10
'''

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation


NUM_CLASSES = 200 
def model(images):
  
  x = Conv2D(32, (3, 3), padding='same')(images)
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
  logits = Dense(NUM_CLASSES)(x)
  
  return logits

    
  

