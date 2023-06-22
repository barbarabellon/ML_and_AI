#!/usr/bin/env python
# coding: utf-8

# # Evaluación Minería de datos II. Ejericicio 2.
# 
# Se elige el segundo conjunto de imágenes es una construcción artificial de personas con y sin mascarilla. La base de datos ha sido obtenida de la siguiente dirección de internet: https://github.com/prajnasb/observations . Hay
# que descargar toda la información entera (https://github.com/prajnasb/observations/archive/master.zip URL de la descarga directa). Posteriormente tenemos que acceder a la carpeta observations-master\experiements\dest_folder donde tendremos las carpetas con las imágenes de entrenamiento, train, validación, val y test, test. Cogemos estas tres carpetas y las llevamos a nuestro directorio de trabajo para trabajar con ellas.
# 
# Cada una de estas carpetas contendrá una carpeta llamada with_mask y otra denominada without_mask, que contendrán las imágenes que tienen mascarilla y las que no. El conjunto de datos se reparte de la siguiente manera:
# - train with_mask: 658
# - train without_mask: 658
# - val with_mask: 71
# - val without_mask: 71
# - test with_mask: 97
# - test without_mask: 97
# 
# Una vez elegido uno de los dos conjuntos de datos se solicita que se realice una clasificación de las imágenes que incluya cualquier estrategia para aumentar en lo posible la precisión: Data Aumentation, Regularización, BatchNormalization, Features Extraction, Fine Tune, modelos preentrenados.... Se valorará que el código esté comentado viendo que se entiende los mecanismos que se están aplicando para conseguir una mejor solución.
# 
# ## Importar librerías y datos

# In[1]:


import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Poner estos comandos antes de ejecutar un modelo de Keras para usar la GPU

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
 # from tensorflow.compat.v1.keras import backend as K
# K.tensorflow_backend._get_available_gpus()


# Poner estos comandos antes de ejecutar un modelo de Keras para usar la GPU
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto( device_count = {'GPU':1,'CPU': 4} ) 
sess = tf.Session(config=config) 
tf.keras.backend.set_session(sess)


from tensorflow.keras.preprocessing import image

base_dir = '/home/barbara/OneDrive/Mineria II/Evaluacion/observations-master/experiements/dest_folder'
print('Con mascarilla')
image.load_img('/home/barbara/OneDrive/Mineria II/Evaluacion/observations-master/experiements/dest_folder/test/with_mask/1-with-mask.jpg')


# In[2]:


#Directorio de trabajo

base_dir = '/home/barbara/OneDrive/Mineria II/Evaluacion/observations-master/experiements/dest_folder'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
# Directorio con las imagenes de training
train_mask_dir = os.path.join(train_dir, 'with_mask')
train_nomask_dir = os.path.join(train_dir, 'without_mask')
# Directorio con las imagenes de validation
validation_mask_dir = os.path.join(validation_dir, 'with_mask')
validation_nomask_dir = os.path.join(validation_dir, 'without_mask')
# Directorio con las imagenes de test
test_mask_dir = os.path.join(test_dir, 'with_mask')
test_nomask_dir = os.path.join(test_dir, 'without_mask')


# In[3]:


train_mask_fnames = os.listdir( train_mask_dir )
train_nomask_fnames = os.listdir( train_nomask_dir )

validation_mask_fnames = os.listdir( validation_mask_dir )
validation_nosmask_fnames = os.listdir( validation_nomask_dir )

test_mask_fnames = os.listdir( test_mask_dir )
test_nomask_fnames = os.listdir( test_nomask_dir )

print('total training mask images :', len(os.listdir(train_mask_dir ) ))
print('total training no mask images :', len(os.listdir(train_nomask_dir ) ))
print('total validation mask images :', len(os.listdir( validation_mask_dir ) ))
print('total validation no mask images :', len(os.listdir( validation_nomask_dir ) ))
print('total test mask images :', len(os.listdir( test_mask_dir ) ))
print('total test no mask images :', len(os.listdir( test_nomask_dir ) ))


# In[4]:


def print_pictures(dir, fnames):
# presentaremos imágenes en una configuración de 4x4
    nrows = 2
    ncols = 2
    pic_index = 0 # Índice para iterar sobre las imagenes
    fig = plt.figure()
    fig.set_size_inches(ncols*2, nrows*2)
    pic_index+=8
    next_pix = [os.path.join(dir, fname) for fname in fnames[pic_index-4:pic_index]]
    for i, img_path in enumerate(next_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.show()


# In[5]:


print_pictures(train_mask_dir,train_mask_fnames)
print_pictures(train_nomask_dir,train_nomask_fnames)


# In[ ]:





# ## Red convolucional

# In[6]:


#librerías
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from numpy import mean
from numpy import std


# ### Arquitectura del modelo
# Es una red convolucional CNN, con la siguiente configuración:
# - Capa Conv2d, con 32 unidades y filtrado 3x3, la función de acutivación relu y padding de tipo "same", para que la salida tenga la misma dimensión que la entrada.
# - max pooling (2x2) o submuestreo de 2x2, max pooling coge el máximo de una matriz 2x2 y lo utiliza en la nueva capa.
# - Segunda Conv2d capa, con 64 unidades y filtrado 3x3, la función de acutivación relu y padding de tipo "same", para que la salida tenga la misma dimensión que la entrada.
# - max pooling (2x2)
# - Tercera Conv2d capa, con 128 unidades y filtrado 3x3, la función de acutivación relu y padding de tipo "same", para que la salida tenga la misma dimensión que la entrada.
# - max pooling (2x2)
# - Cuarta Conv2d capa, con 128 unidades y filtrado 3x3, la función de acutivación relu y padding de tipo "same", para que la salida tenga la misma dimensión que la entrada.
# - max pooling (2x2)
# - Dense layer, with 128 units
# - Dense sigmoid layer
# - On compilation, use adam as the optimizer and categorical_crossentropy as the loss function. Add 'accuracy' as a metric

# In[8]:


#Definir el modelo
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compilar el model
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'])
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=base_dir,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)


# In[ ]:


# def define_model():
#     # define model
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
#                      kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization())
#     model.add(Dense(10, activation='softmax'))
#     # compile model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# In[11]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale=1./255, 
                                   rotation_range=40,
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator( rescale = 1.0/255.)
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size=20, 
                                                    class_mode='binary', 
                                                    target_size=(150, 150))
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=20, 
                                                              class_mode = 'binary', 
                                                              target_size = (150, 150))
test_generator = test_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode = 'binary', 
                                                  target_size = (150, 150))


# In[ ]:


batch_size = 32
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size
history = model.fit(train_generator,
                    steps_per_epoch= steps_per_epoch,
                    epochs= 50,
                    validation_data= validation_generator,
                    validation_steps= validation_steps,
                    verbose=1,
                    callbacks=[model_checkpoint_callback])


# We are going to train small steps of our model in order to evaluate the hyperparameters and the strategy. In order to do that, we will define a step epochs number and will train and evaluate the model for that amount of epochs. after a number of repeats we will reduce the effect random initialization of certain parameters.
# 
# <font color=red>Evaluate the built model by training 10 times on different initializations<b> Hint: we would like to have some parameters of the score distribution, like the ones imported  </b></font>

# In[ ]:


# step_epochs = 3
# batch_size = 128

# def evaluate_model(model, trainX, trainY, testX, testY):
#     # fit model
#     model.fit(trainX, trainY, epochs=step_epochs, batch_size=batch_size, verbose=1)
#     # evaluate model
#     _, acc = model.evaluate(testX, testY, verbose=0)
#     return acc


# def evaluate(trainX, trainY, testX, testY, repeats=10):
#     scores = list()
#     for _ in range(repeats):
#         # define model
#         model = define_model()
#         # fit and evaluate model
#         accuracy = evaluate_model(model, trainX, trainY, testX, testY)
#         # store score
#         scores.append(accuracy)
#         print('> %.3f' % accuracy)
#     return scores


# # evaluate model
# scores = evaluate(trainX, trainY, testX, testY)
# # summarize result
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# # ### Keras Image data generator
# # In order to perform some data augmentation, Keras includes the Image data generator, which can be used to improve performance and reduce generalization error when training neural network models for computer vision problems. 
# # A range of techniques are supported, as well as pixel scaling methods. Some of the most commn are: 
# # 
# # - Image shifts via the width_shift_range and height_shift_range arguments.
# # - Image flips via the horizontal_flip and vertical_flip arguments.
# # - Image rotations via the rotation_range argument
# # - Image brightness via the brightness_range argument.
# # - Image zoom via the zoom_range argument.
# # 
# # 
# # Let's see it with an example:
# # 

# # In[ ]:


# from numpy import expand_dims
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # In[ ]:


# # expand dimension to one sample
# from numpy import expand_dims
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# data = trainX[0]*255
# samples = expand_dims(data, 0)
# # create image data augmentation generator
# datagen = ImageDataGenerator(horizontal_flip=True, 
#                              featurewise_center=True,
#                              featurewise_std_normalization=True,
#                              rotation_range=20,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2)

# # prepare iterator
# it = datagen.flow(samples, batch_size=1)
# # generate samples and plot
# for i in range(9):
#     # define subplot
#     plt.subplot(330 + 1 + i)
#     # generate batch of images
#     batch = it.next()
#     # convert to unsigned integers for viewing
#     image = batch[0].astype('uint8')
#     # plot raw pixel data
#     plt.imshow(image)
# # show the figure
# plt.show()


# # In[ ]:


# batch.shape


# # <font color=red>Evaluate the model with data augmentation <br> Hint: Use the ?model.fit_generator command and please take into acount the parameters of the model.fit_generator: It needs to include: epochs, steps_per_epoch and a generator (i.e: a flow of images). </font>

# # In[ ]:


# # fit and evaluate a defined model
# def evaluate_model_increased(model, trainX, trainY, testX, testY):
#     datagen = ImageDataGenerator(horizontal_flip=True)
#     # in case there is mean/std to normalize
#     datagen.fit(trainX)

#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(trainX, trainY,
#                                      batch_size=batch_size),
#                         epochs=step_epochs,
#                         steps_per_epoch=len(trainX) // batch_size,
#                         verbose =1)
    
#     # evaluate model
#     _, acc = model.evaluate(testX, testY, verbose=1)
#     return acc

# # repeatedly evaluate model, return distribution of scores
# def repeated_evaluation_increased(trainX, trainY, testX, testY, repeats=10):
#     scores = list()
#     for _ in range(repeats):
#         # define model
#         model = define_model()
#         # fit and evaluate model
#         accuracy = evaluate_model_increased(model, trainX, trainY, testX, testY)
#         # store score
#         scores.append(accuracy)
#         print('> %.3f' % accuracy)
#     return scores

# # evaluate model
# scores = repeated_evaluation_increased(trainX, trainY, testX, testY)
# # summarize result
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# # ### Test Time augmentation (TTA)
# # 
# # The image data augmentation technique can also be applied when making predictions with a fit model in order to allow the model to make predictions for multiple different versions of each image in the test dataset. Specifically, it involves creating multiple augmented copies of each image in the test set, having the model make a prediction for each, then returning an ensemble of those predictions.(e.g: majority voting in case of classification)
# # 
# # Augmentations are chosen to give the model the best opportunity for correctly classifying a given image, and the number of copies of an image for which a model must make a prediction is often small, such as less than 10 or 20. Often, a single simple test-time augmentation is performed, such as a shift, crop, or image flip.

# # <font color=red>Evaluate the model with data augmentation. <b>Please note on this case we are not going to use the generateor on training, but on testing.</b> <br> Hint: Use the model.predict_generator function </font>

# # In[ ]:


# from sklearn.metrics import accuracy_score
# import numpy as np
# n_examples_per_image = 3


# # make a prediction using test-time augmentation
# def prediction_augmented_on_test(datagen, model, image, n_examples):
#     # convert image into dataset
#     samples = expand_dims(image, 0)
#     # prepare iterator
#     it = datagen.flow(samples, batch_size=n_examples)
#     # make predictions for each augmented image
#     yhats = model.predict_generator(it, steps=n_examples, verbose=0)
#     # sum across predictions
#     summed = np.sum(yhats, axis=0)
#     # argmax across classes
#     return np.argmax(summed)

# # evaluate a model on a dataset using test-time augmentation
# def evaluate_model_test_time_agumented(model, testX, testY):
#     # configure image data augmentation
#     datagen = ImageDataGenerator(horizontal_flip=True)
#     # define the number of augmented images to generate per test set image
#     yhats = list()
#     for i in range(len(testX)):
#         # make augmented prediction
#         yhat = prediction_augmented_on_test(datagen, model, testX[i], n_examples_per_image)
#         # store for evaluation
#         yhats.append(yhat)
#     # calculate accuracy
#     testY_labels = np.argmax(testY, axis=1)
#     acc = accuracy_score(testY_labels, yhats)
#     return acc

# def evaluate_model_test_augmented(model, trainX, trainY, testX, testY):
#     # fit model
#     model.fit(trainX, trainY, epochs=step_epochs, batch_size=batch_size, verbose=0)
#     # evaluate model
#     acc = evaluate_model_test_time_agumented(model, testX, testY)
#     return acc

# def evaluate_test_augmented(trainX, trainY, testX, testY, repeats=10):
#     scores = list()
#     for _ in range(repeats):
#         # define model
#         model = define_model()
#         # fit and evaluate model
#         accuracy = evaluate_model_test_augmented(model, trainX, trainY, testX, testY)
#         # store score
#         scores.append(accuracy)
#         print('> %.3f' % accuracy)
#     return scores


# # evaluate model
# scores = evaluate_test_augmented(trainX, trainY, testX, testY)
# # summarize result
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# # In[ ]:




