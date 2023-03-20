from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing  import image
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import h5py



listas = glob.glob('IMAGENS ANEMIA (Padronizado)/*')
Image.open(listas[1])


def myTarget(imagem):
    imgm = float(imagem.split('/')[-1].split('__')[1].split('_')[1])
    return imgm

target_list = []
for img_path in listas:

    if myTarget(img_path) >= 12:
        target_list.append(1)

    else:
        target_list.append(0)

target_list_array = np.array(target_list)

target_list_array

train_images = []

for img in listas:
    i = image.load_img(img, target_size=(224,224))
    i = image.img_to_array(i)
    train_images.append(i)

train_images = np.array(train_images)

train_data = train_images.reshape(train_images.shape[0],150528)

train_data = train_data/train_data.max()

X_train, X_test, y_train, y_test = train_test_split(train_data, target_list_array, test_size=0.20, random_state=0)

#Encoder
# define encoder

n_inputs = 200
n_inputstrue = X_train.shape[1]
print(n_inputs)
print(n_inputstrue)

visible = Input(shape=(n_inputstrue,))

# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)


# bottleneck (Gargalo)
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

#level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# output layer
output = Dense(n_inputstrue, activation='linear')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

model.summary()

model.compile(optimizer='adam',loss='mse')
model.fit(X_train, X_train, epochs=3, batch_size=31, verbose=0, validation_data=(X_test,X_test))

model.save('saved_models/train_anemia_2epochs')


