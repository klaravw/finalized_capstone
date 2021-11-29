from posixpath import split
import sys
import pandas as pd
import numpy as np 
import os
import shutil as sh
from sklearn.model_selection import train_test_split
from PIL import Image

# RGB chips have an alpha channel

# Global Variables
SET_SIZE = int(sys.argv[1])    # Number of rows to extract into subset
VAL_SET_SIZE = 201

READ_PATH = "/projects/cmda_capstone_2021_ti/data/data_summary_final_summary.csv"   # Path of csv file to read from
WRITE_PATH = "/projects/cmda_capstone_2021_ti/data/training_sets/"    # Path of directory where output training csv file is to be saved

NCHIP_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/NativeChips/"
RGBCHIP_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/ColorChips/"
FIVE_MASK_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/05masks/"

NCHIP_SUBPATH = "NativeChips"
RGBCHIP_SUBPATH = "ColorChips"
FIVE_MASK_SUBPATH = "05masks"


summary = pd.read_csv(READ_PATH)
#print(summary.head(3))
print("Full data organizer loaded.")

cols = ['05min_Lightning_Count', '15min_Lightning_Count','30min_Lightning_Count']
summary[cols] = summary[cols].apply(pd.to_numeric, errors='coerce')

# Sort set by date
summary.sort_values(by='EpochTime', ascending=False)

# Write training set to file
training_set = summary.head(SET_SIZE)
training_set.to_csv(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv", index=False)
print(WRITE_PATH + "trainingset_descending_" + str(SET_SIZE) + ".csv" + "created with " + str(SET_SIZE) + "records!")

#------------------------------------------------
# Creating Train and validation sub-folders
TRAIN_WRITE_PATH = os.path.join(WRITE_PATH, "Train/")
VAL_WRITE_PATH = os.path.join(WRITE_PATH, "Val/")
TEST_WRITE_PATH = os.path.join(WRITE_PATH, "Test/")
train_dir = os.mkdir(TRAIN_WRITE_PATH)
val_dir = os.mkdir(VAL_WRITE_PATH)
test_dir = os.mkdir(TEST_WRITE_PATH)

# Creating directory for training images
nchip_path_t = os.path.join(TRAIN_WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path_t = os.path.join(TRAIN_WRITE_PATH, RGBCHIP_SUBPATH)
five_path_t = os.path.join(TRAIN_WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir_t = os.mkdir(nchip_path_t)
rgbchip_dir_t = os.mkdir(rgbchip_path_t)
fivemask_dir_t = os.mkdir(five_path_t)

nchip_path_v = os.path.join(VAL_WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path_v = os.path.join(VAL_WRITE_PATH, RGBCHIP_SUBPATH)
five_path_v = os.path.join(VAL_WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir_v = os.mkdir(nchip_path_v)
rgbchip_dir_v = os.mkdir(rgbchip_path_v)
fivemask_dir_v = os.mkdir(five_path_v)

nchip_path_te = os.path.join(TEST_WRITE_PATH, NCHIP_SUBPATH)
rgbchip_path_te = os.path.join(TEST_WRITE_PATH, RGBCHIP_SUBPATH)
five_path_te = os.path.join(TEST_WRITE_PATH, FIVE_MASK_SUBPATH)
nchip_dir_te = os.mkdir(nchip_path_te)
rgbchip_dir_te = os.mkdir(rgbchip_path_te)
fivemask_dir_te = os.mkdir(five_path_te)

print("All subfolders created succesfully")

#------------------------------------------------
#Copying files from main folders to trainingset subfolders

# Create split objects for iteration
tvt_split=[0.50,0.25,0.25]
train_split,test_split = train_test_split(training_set,test_size=tvt_split[0],train_size=sum(tvt_split[1:]),shuffle=False)
val_split,test_split = train_test_split(test_split,test_size=split[1]*2,train_size=split[2]*2,shuffle=False)

train_split_zip = zip(train_split["Colorized_Chip_Name"], train_split["Native_Chip_Name"], train_split["05min_Mask_Name"])
val_split_zip = zip(val_split["Colorized_Chip_Name"], val_split["Native_Chip_Name"], val_split["05min_Mask_Name"])
test_split_zip = zip(test_split["Colorized_Chip_Name"], test_split["Native_Chip_Name"], test_split["05min_Mask_Name"])

record_loss = 0 # Records how many files are missing
for color, native, mask05 in train_split_zip:
    if (color != "None") and (native != "None") and (mask05 != "None"):
        sh.copy(NCHIP_READ_PATH + native, nchip_path_t)
        sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path_t)
        sh.copy(FIVE_MASK_READ_PATH + mask05, five_path_t)
    else:
        record_loss += 1

for color, native, mask05 in val_split_zip:
    if (color != "None") and (native != "None") and (mask05 != "None"):
        sh.copy(NCHIP_READ_PATH + native, nchip_path_v)
        sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path_v)
        sh.copy(FIVE_MASK_READ_PATH + mask05, five_path_v)
    else:
        record_loss += 1

for color, native, mask05 in test_split_zip:
    if (color != "None") and (native != "None") and (mask05 != "None"):
        sh.copy(NCHIP_READ_PATH + native, nchip_path_te)
        sh.copy(RGBCHIP_READ_PATH + color, rgbchip_path_te)
        sh.copy(FIVE_MASK_READ_PATH + mask05, five_path_te)
    else:
        record_loss += 1



# Data augmentation functions
#TODO Data Augmentation
# Need to implement functions that allow us to create augmented data records such as flipping and rotating images
# Can most likely be done with the PIL rotate(degree) function
def create_augmented_record(mode=0):
    pass



if record_loss == 0:
    print("All images copied to subfolder succesfully!")
else:
    print("Images copied to directories succesfully, " + str(record_loss) + " records were missing and could not be copied.")


print("Executed succesfully, go train that model stud!")