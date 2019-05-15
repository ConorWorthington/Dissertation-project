from sklearn.model_selection import train_test_split
from keras import *
from keras.utils import plot_model
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np


 #Load external data source and process so as to return faces and expected emotional state
def loadDataset():
    data = pd.read_csv(datasetPath)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),(48,48))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions
 
def processFaces(x):
    x = x.astype('float32')
    x = x / 255.0
    x = x - 0.5
    x = x * 2.0
    return x

#Ensures it's running on GPU
print(len(backend.tensorflow_backend._get_available_gpus()))
datasetPath = 'E:/new downloads/Comp project - final year/Emotion detection\challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv' #custom path to data
faces, emotions = loadDataset()
faces = processFaces(faces) #Convert to expected scale to learn from
xTrain, xTest,yTrain,yTest = train_test_split(faces, emotions,test_size=0.2,shuffle=True) #Split up data source into training and test data

 
#Build up model 
model = Sequential()
model.add(BatchNormalization(input_shape=(48,48,1)))
model.add(Convolution2D(96, 4, 4, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Conv2D(128, 2, 2,init='he_normal',activation='relu'))
model.add(Conv2D(128, 2, 2,init='he_normal',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(Conv2D(128, 3, 3,init='he_normal',activation='relu'))
model.add(Conv2D(128, 3, 3,init='he_normal',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
model.add(GlobalAveragePooling2D());
model.add(Dense(32, activation='relu'))
model.add(Dense(7,activation='softmax',name='predictions'))
adams = optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
model.compile(optimizer=adams, loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 50 #Set epochs

 #Train dataset
model.fit(xTrain,yTrain,validation_split=0.0,validation_data=(xTest,yTest), shuffle=True, epochs=epochs, batch_size=20, verbose=1)
model.save('emotion_model.h5') #Save dataset
plt.plot(history.history['acc']) #Display plots
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
del model;#Reset model
model = load_model('emotion_model.h5')#Ensure model has saved
img = cv2.imread('willTest.jpg')#Test on real data
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = (gray / 255)
roi_rescaled = cv2.resize(gray,(48,48))
predictions = model.predict(roi_rescaled[np.newaxis, :, :, np.newaxis])
print(predictions)
img = cv2.imread('niallTest.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = (gray / 255)
roi_rescaled = cv2.resize(gray,(48,48))
predictions = model.predict(roi_rescaled[np.newaxis, :, :, np.newaxis])
print(predictions)
