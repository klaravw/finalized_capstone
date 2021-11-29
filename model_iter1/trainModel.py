from model import *
import pandas as pd
import numpy as np
from matplotlib import image
from matplotlib import pyplot
from dataLoader import *
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img


# Load Training/Testing Data
#native_chips, _05masks = load_data("/projects/cmda_capstone_2021_ti/data/training_sets/trainingset_descending_40.csv")

im = load_img("/projects/cmda_capstone_2021_ti/data/Data/NativeChips/Chip_Native_B200_OR_ABI-L2-CMIPC-M3C13_G16_s20171912142189_e20171912144574_c20171912145010_EP552995097.411295.png", color_mode = "grayscale")
img_array = img_to_array(im)

# Construct model
model = unet()


# Fit Model
preds = model.predict(img_array.reshape([-1,400, 400,1]))
asdfg = array_to_img(preds[0])
save_img("model_IO_test_output.png", preds[0])

# Save Model