import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from tensorflow.python import keras
from keras import *
from keras.layers import *
import cv2
from keras.models import *


#Parses from text to numpy array
def string2image(string):
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

#Ensures it's running on GPU
print(len(backend.tensorflow_backend._get_available_gpus()))
print(keras.__version__)
data = pd.read_csv('E:/new downloads/Comp project - final year/facial-keypoints-detection/training/training.csv')
keypoint_cols = list(data.columns)[:-1]
fully_annotated = data.dropna() #Removes useless data with fewer than 15 points - this could harm training if data points initialised as null
x = np.stack([string2image(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]
y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)
xTrain, xTest,yTrain,yTest = train_test_split(x/255, y,test_size=0.2,shuffle=True) #Split up data into both test and training arrays 



#Build model 
model = Sequential()
model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Convolution2D(96, 4, 4, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Conv2D(128, 2, 2,init='he_normal',activation='relu'))
model.add(Conv2D(128, 2, 2,init='he_normal',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Conv2D(128, 3, 3,init='he_normal',activation='relu'))
model.add(Conv2D(128, 3, 3,init='he_normal',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Conv2D(256, 5, 5,init='he_normal',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(GlobalAveragePooling2D());
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30))
adams = optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False) #Select optimizer 
model.compile(optimizer=adams, loss='mse', metrics=['accuracy'])
epochs = 250 #Set number of epochs
history = model.fit(xTrain, yTrain, validation_split=0.0,validation_data=(xTest,yTest), shuffle=True, epochs=epochs, batch_size=20, verbose=1) #Begin training
model.save('keypoints_model.h5') #Save model
plt.plot(history.history['acc']) #Plot model
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
del model #Delete model 
model = load_model('keypoints_model.h5') #Ensure model has loaded and training works
img = cv2.imread('niallTest.jpg') #Test model with real data
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = (gray / 255)
roi_rescaled = cv2.resize(gray,(96,96))
predictions = model.predict(roi_rescaled[np.newaxis, :, :, np.newaxis])
xy_predictions = (predictions).reshape(15, 2)
plt.imshow(roi_rescaled, cmap='gray');
plt.plot(xy_predictions[:, 0], xy_predictions[:, 1], 'b*')
plt.show()
print('stop')