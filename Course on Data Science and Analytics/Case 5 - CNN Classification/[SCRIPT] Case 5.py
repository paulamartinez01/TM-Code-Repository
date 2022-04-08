# LOADING THE DATASET
from tensorflow.keras.datasets import fashion_mnist
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow 

#%% INITIAL VISUALIZATION
index = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(8,9))
k = 1 
for j in np.random.randint(0,500,16):
    plt.subplot(4,4,k)
    k += 1
    plt.imshow(trainX[j], cmap='autumn')
    plt.title('{} / {}'.format(index[trainY[j]], trainY[j]))
    plt.axis('off')
   
#%% CHECKING FOR THE VALUES OF Y
print(trainY) # The values are from 0-9

#%% making trainY and testY categorical
from tensorflow.keras.utils import to_categorical as to_cat
trainY_cat = to_cat(trainY)
testY_cat = to_cat(testY)

print(trainY_cat)
print(testY_cat)
    
#%% CHECKING FOR THE CONSISTENCY IN SHAPE
print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

#%% WHAT IS THE CURRENT DIMENSION OF THE IMAGE?
print(trainX.ndim, trainX.shape)

#%% ADDING 1 CHANNEL TO Xs
trainX = trainX.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])
print(trainX.shape)

#%% NORMALIZING THE IMAGES (scaling to have values of 0 and 1)
trainX_norm = trainX / 255
testX_norm = testX / 255

#%% SPLITTING TRAIN FURTHER INTO TRAINING AND VALIDATION DATASETS
from sklearn.model_selection import train_test_split
trainX2, trainY2, valX, valY = train_test_split(trainX, trainY_cat, test_size=0.2, random_state=2018)  

#%% MODELLING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.python import keras

model = keras.models.Sequential([
keras.layers.Conv2D(filters = 32 , kernel_size = 3,strides = (1,1), padding = 'valid',activation = 'relu',input_shape = [28,28,1]), # 1st Layer


keras.layers.Conv2D(filters = 64 , kernel_size = 3,strides = (1,1), padding = 'valid',activation = 'relu',input_shape = [28,28,1]), # 2nd Layer
keras.layers.MaxPooling2D(pool_size = (2,2)),

keras.layers.Dropout(0.5),

keras.layers.Conv2D(filters = 128 , kernel_size = 3,strides = (1,1), padding = 'valid',activation = 'relu',input_shape = [28,28,1]), # 3rd Layer
keras.layers.MaxPooling2D(pool_size = (2,2)),

keras.layers.Dropout(0.5),

keras.layers.Flatten(),

keras.layers.Dense(units = 128,activation = 'relu'),

keras.layers.Dense(units = 10,activation = 'softmax')

])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
print(model.summary())
#%% CREATING AN EARLY STOP
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)
results = model.fit(trainX, trainY_cat, epochs=20, validation_data=(testX,testY_cat), callbacks=[early_stop])

#%% FLATTENING THE DATA
model.metrics_names
losses = pd.DataFrame(results.history)
losses.head()
#%%
losses[['accuracy', 'val_accuracy']].plot()
losses[['loss','val_loss']].plot()
print(model.metrics_names)

#%%
print(model.evaluate(testX, testY_cat, verbose=0))

#%%
model_loss = pd.DataFrame(results.history)
model_loss.plot()

#%% USING THE MODEL FOR PREDICTION
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(testX)
predictions = np.argmax(predictions, axis=1)
testY_cat.shape

print(classification_report(testY, predictions))
confusion_matrix(testY, predictions)

#%% testing
rows = [0,1,2,3,4,5,6,7,8,9]
k = 1 
for i in rows:
    my_image = testX[i]
    plt.imshow(my_image.reshape(28,28), cmap='autumn')

#%% testing 2 with predictions
index = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
rows = [0,1,2,3,4,5,6,7,8,9]
k = 1 
plt.figure(figsize=(8,9))
for i in rows:
    my_image = testX[i]
    plt.subplot(5,5,k)
    k += 1
    plt.imshow(my_image, cmap='autumn')
    myimage_prediction = model.predict(my_image.reshape(1,28,28,1))
    myimage_prediction = np.argmax(myimage_prediction, axis=1)
    print(myimage_prediction)
   # plt.title('{}'.format(myimage_prediction)
    plt.axis('off')
#%%
my_image = testX[1]
plt.imshow(my_image.reshape(28,28), cmap='autumn')
myimage_prediction = model.predict(my_image.reshape(1,28,28,1))
myimage_prediction = np.argmax(myimage_prediction, axis=1)
print(myimage_prediction)
