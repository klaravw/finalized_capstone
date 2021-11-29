from model import *
import pandas as pd
import numpy as np
from matplotlib import image, interactive
from matplotlib import pyplot
from dataLoader import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import os
import shutil as sh
import errno
import matplotlib.pyplot as plt

#os environ gpu
# Load Training/Testing Data
#native_chips, _05masks = load_data("/projects/cmda_capstone_2021_ti/data/training_sets/trainingset_descending_500.csv")

parent_data_dir = '/projects/cmda_capstone_2021_ti/data/training_sets'
img_size=(400,400)
batch_size = 9

#Create data generators
train_input_image_paths,train_target_image_paths = get_data_paths(parent_data_dir,'Train')
val_input_image_paths,val_target_image_paths = get_data_paths(parent_data_dir,'Val')
train_gen = DataGather(batch_size, img_size, train_input_image_paths, train_target_image_paths,shuffle=True)
val_gen = DataGather(batch_size, img_size, val_input_image_paths, val_target_image_paths,shuffle=True)
print(len(train_gen))
# data_gen_args = dict(rotation_range=0,
#                     width_shift_range=0,
#                     height_shift_range=0,
#                     shear_range=0,
#                     zoom_range=0,
#                     horizontal_flip=False,
#                     fill_mode='nearest')
# myGene = trainGenerator(4,'/projects/cmda_capstone_2021_ti/data/training_sets','NativeChips','05masks',data_gen_args,save_to_dir = None)
# Construct model
model = unet()

mc = ModelCheckpoint(mode='min', filepath='model_iter1/model_accuracy/top-weights.h5', monitor='val_loss',save_best_only='True', save_weights_only='False', verbose=1)
es = EarlyStopping(mode='min', monitor='val_loss', patience=25, verbose=0)
rl = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1,mode="min",min_lr=0.000000001)
cv = CSVLogger("model_iter1/model_accuracy/log.csv" , append=True , separator=',')
callbacks = [mc, rl, cv, es]

# Fit Model
history = model.fit(train_gen, epochs=40, validation_data=val_gen, callbacks=callbacks) # steps_per_epoch=len(train_gen)/batch_size

try:
    os.mkdir("model_iter1/model_accuracy")
except OSError as e:
    # if e.errno == e.EEXIST:
    sh.rmtree("model_iter1/model_accuracy")
    os.mkdir("model_iter1/model_accuracy")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
fig,ax = plt.subplots(1)
ax.plot(train_loss,label='train_loss')
ax.plot(val_loss,label='val_loss')
plt.legend()
plt.savefig('model_iter1/model_accuracy/Loss_Curves.png')
plt.close()
    
train_recall = history.history['recall']
train_precision = history.history['precision']
val_recall = history.history['val_recall']
val_precision = history.history['val_precision']
    
fig,ax = plt.subplots(1)
ax.plot(train_recall,label='train_recall',color='red',linestyle='-')
ax.plot(val_recall,label='val_recall',color='red',linestyle='--')
ax.plot(train_precision,label='train_precision',color='blue',linestyle='-')
ax.plot(val_precision,label='val_precision',color='blue',linestyle='--')
plt.legend()
plt.savefig('model_iter1/model_accuracy/Precision_Recall.png')
plt.close()

# plt.plot(history.history["accuracy"])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.savefig("model_iter1/model_accuracy/Accuracy_Plot.png")

pred = model.predict(val_gen[150])
sample_image = train_gen[150][0][0]
sample_gt = train_gen[150][1][0][:,:,0]
pred = pred[0,:,:,0]
fig,ax = plt.subplots(1,3)
ax[0].imshow(sample_image)
ax[1].imshow(sample_gt)
ax[2].imshow((pred>0.5)*1)
plt.savefig('model_iter1/model_accuracy/Val_Sample.png')
plt.close()

# preds = model.predict(native_chips[0].reshape([-1,400, 400,1]))
# pred_img = preds[0]
# true_chip = native_chips[0]
# true_mask = _05masks[0]
# save_img("model_iter1/model_accuracy/trained_model_sample_pred.png", pred_img) # Show image and look at probab vals
# save_img("model_iter1/model_accuracy/Predicted_native_chip.png", true_chip)
# save_img("model_iter1/model_accuracy/True_Mask.png", true_mask)

# # Save Model
# model.save("model_iter1/saved_model")
