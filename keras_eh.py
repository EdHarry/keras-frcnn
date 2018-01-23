
# coding: utf-8

# In[1]:


import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

from keras import backend as bkend
bkend.set_session(sess)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GRU
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from PIL import Image, ImageDraw


# In[2]:


def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=5,
		 nbFilters=128, filtersize = 3):


	model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):
    
    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
		   
        #conv1 = Convolution2D(out_num_filters, 1, 1, border_mode='same')
        conv1 = Conv2D(out_num_filters, (1, 1), padding='same')
        stack1 = conv1(l)   	
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)
        
        #conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        conv2 = Conv2D(out_num_filters, (filtersize, filtersize), kernel_initializer='he_normal', padding='same')
        stack4 = conv2(stack3)
        #stack5 = merge([stack1, stack4], mode='sum')
        stack5 = Add()([stack1, stack4])
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)
    	
        #conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        conv3 = Conv2D(out_num_filters, (filtersize, filtersize), padding='same')
        #conv3.set_weights(conv2.get_weights())
        stack8 = conv3(stack7)
        #stack9 = merge([stack1, stack8], mode='sum')
        stack9 = Add()([stack1, stack8])
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)    
        
        #conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        conv4 = Conv2D(out_num_filters, (filtersize, filtersize), padding='same')
        #conv4.set_weights(conv2.get_weights())
        stack12 = conv4(stack11)
        #stack13 = merge([stack1, stack12], mode='sum')
        stack13 = Add()([stack1, stack12])
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)    
        
        if pool:
            stack16 = MaxPooling2D((2, 2), padding='same')(stack15) 
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)
            
        return stack17

    #Build Network
    input_img = Input(shape=(shape1, shape2, nbChannels))
    #conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    conv_l = Conv2D(nbFilters, (filtersize, filtersize), padding='same', activation='relu')
    l = conv_l(input_img)
    
    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)
    
    out = Flatten()(l)        
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(32, activation='relu')(out)
    l_out = Dense(nbClasses, activation = 'softmax')(out)
    
    model = Model(inputs=input_img, outputs=l_out)
    
    return model


# In[3]:


def MakeCellImage(x, y, r, i):
    im = Image.new(mode='F', size=(128, 128))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy=[x-r, y-r, x+r, y+r], fill='White')
    im = np.array(im).astype(np.float32)
    im *= (i / 255.0)
    return im

def MakeRandomCellImage():
    radius = np.random.randint(low=-5, high=10) + 10
    intensity = (np.random.randn() * 0.1) + 0.5
    intensity = max(min(intensity, 1.0), 0.0)
    position = np.random.randint(low=radius, high=128-radius, size=2)
    im = MakeCellImage(position[0], position[1], radius, intensity)
    return im, np.array([position[0], position[1], intensity])

def MakeCellTrainingData(n):
    x_train = np.zeros(shape=(n, 128, 128, 1))
    y_train = np.zeros(shape=(n, 3))
    for index in range(n):
        im, data = MakeRandomCellImage()
        x_train[index, :, :, :] = im[:, :, np.newaxis]
        y_train[index, :] = data[np.newaxis, :]
    return x_train, y_train


# In[4]:


model_rcnn = makeModel(1, 128, 128, 2)


# In[4]:


model_conv = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model_conv.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model_conv.add(Conv2D(32, (3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(Dropout(0.25))

model_conv.add(Conv2D(64, (3, 3), activation='relu'))
model_conv.add(Conv2D(64, (3, 3), activation='relu'))
model_conv.add(MaxPooling2D(pool_size=(2, 2)))
model_conv.add(Dropout(0.25))

model_conv.add(Flatten())
model_conv.add(Dense(256, activation='relu'))
model_conv.add(Dropout(0.5))
model_conv.add(Dense(32, activation='relu'))
model_conv.add(Dense(2, activation='softplus'))


# In[5]:


def AddNoise(im):
    im += ((np.random.randn(im.shape[0], im.shape[1], im.shape[2]) * 0.1) + 0.2)
    im[im < 0] = 0
    im[im > 1] = 1
    return im


# In[67]:


model_conv.compile(loss='mean_squared_error', optimizer='Nadam')


# In[12]:


model_rcnn.compile(loss='mean_squared_logarithmic_error', optimizer='Nadam')


# In[7]:


def Generator(batch_size):
    x, y = MakeCellTrainingData(batch_size)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=AddNoise)
    datagen.fit(x)
    return datagen.flow(x, y[:, 0:2], batch_size=batch_size)


# In[68]:


model_conv.fit_generator(Generator(64),
                    steps_per_epoch=1, epochs=2**10,
                        workers=8, use_multiprocessing=True)


# In[13]:


early_stopping = EarlyStopping(monitor='loss', patience=2)
model_rcnn.fit_generator(Generator(8),
                    steps_per_epoch=1, epochs=2**14,
                        workers=8, use_multiprocessing=True,
                        callbacks=[early_stopping])


# In[42]:


model_conv.save("/home/ed/Documents/JupyterWorkbooks_tmp/TestModel_conv_wnoise.h5")


# In[69]:


testSize = 32
x_test, y_test = MakeCellTrainingData(testSize)

datagen_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=AddNoise)
datagen_test.fit(x_test)
gen = datagen_test.flow(x_test, y_test[:, 0:2], batch_size=testSize)
nxt = gen.next()
np.concatenate((model_conv.predict(nxt[0]), nxt[1][:, 0:2]), axis=1)


# In[ ]:




