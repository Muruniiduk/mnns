#!/usr/bin/env python3
import pandas as pd
import mujoco_py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, Dense, \
                        Flatten, Dropout, Reshape
from keras.models import Model, Sequential
from keras import optimizers
from keras.regularizers import l1_l2
from keras.initializers import Zeros as initZeros

from keras.models import load_model
from keras.models import model_from_json
import datetime
import sys

def open_and_load(string):
    """Open model and load weights"""
    with open(string+'.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(string+'.h5')
    return model

def load_128_images_starting(a):
    """Loads 128 images starting with a. Loading all images crashes."""
    images = []
    loc = "/home/erik/mujocopy_testikas/frames/"
    for i in range(a,a+128):
        s = ("%.4d" % i) #0200
        images += [cv2.imread(loc+"img_"+s+".png")]
    return images

def save_model(model, model_str):
    """Saves model with weights."""
    model_json = model.to_json()
    with open('models/'+model_str+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('models/'+model_str+".h5")
    print("Saved model to disk, name: ", model_str)

df = pd.read_csv('df.csv')


if (len(sys.argv) == 2):
    #if model is trained in some file already
    model_str = sys.argv[1]
    model = open_and_load(model_str)
    #print("HISTORY!")
    #print(model.get_weights())
    model.compile(optimizer='adam',
                 loss='mean_squared_error')
    print(sys.argv[1])
    #print(model.get_weights())

else:
### Model creation and training
    inputs = Input(shape=(480,480,3))
    conv = Conv2D(filters=1, kernel_size=1, padding="same", name='conv')(inputs)
    # # h = Reshape((480,480))(h)
    h = MaxPooling2D()(conv)
    h = Conv2D(filters=1, kernel_size=3, padding="same")(h)
    h = MaxPooling2D()(h)
    # h = Conv2D(filters=1, kernel_size=3, padding="same")(h)
    # h = MaxPooling2D()(h)
    # # h = Conv2D(filters=1, kernel_size=3, padding="same")(h)
    # h = MaxPooling2D()(h)
    # # h = Conv2D(filters=1, kernel_size=3, padding="same")(h)
    # h = MaxPooling2D()(h)
    h = Flatten()(h)
    # h = Dense(16, activation='linear', kernel_regularizer=l1_l2(l2=0.1),kernel_initializer=initZeros())(h)
    # h = Dense(8, kernel_regularizer=l1_l2(l2=0.1),kernel_initializer=initZeros())(h)
    # h = Dropout(0.33)(h)
    h = Dense(2, activation='linear', kernel_initializer=initZeros())(h) #, kernel_initializer=initZeros()

    model = Model(inputs=inputs, outputs=h)
    model.compile(optimizer='adam',
                 loss='mean_squared_error')


    #Good filter init
    good_filter = np.array([[[[-0.03846514],
             [ 0.02321649],
             [ 0.8732823 ]]]], dtype=np.float32)
    init = [good_filter, np.array([0], dtype=np.float32)]
    model.get_layer(name='conv').set_weights(init)

# Training
losses = []
c = 0
try:
    for j in range(100):
        if j % 10 == 0: print("loop: ", j)
        for i in np.arange(1,8872, 128): #9000+ on testimiseks
            X = np.array(load_128_images_starting(i))[:,:,:,::-1]
            Y = df.iloc[i-1:i+127,:2]
            assert X.shape[0] == Y.shape[0]
            model.fit(X,Y,verbose=False)
            if c % 5 == 1:
                print("avg loss: %.4f" % np.mean(losses), end=' ')
                print("last loss: %.4f" % losses[-1])
            c += 1
            losses += model.history.history['loss']
except KeyboardInterrupt:
    save_model(model, "model_"+str(datetime.datetime.now()).replace(' ','_'))

def test_single_instance(i):
    img = cv2.imread(("/home/erik/mujocopy_testikas/frames/img_%.4d.png" % i))
    X = np.array([img])
    p = model.predict(X)[0]
    y = df.iloc[i][:2]
    print("RMSE on single instance: ", np.sqrt(np.mean((p-y)**2)))

test_single_instance(1)
