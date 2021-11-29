import pandas as pd
import numpy as np
from matplotlib import image
import sys
import os
import random
from skimage import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# File loads in image data given from a csv file containing img paths into a numpy array and returns it.

IMG_READ_PATH = "/projects/cmda_capstone_2021_ti/data/Data/"

# ======= Helper Functions =======
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def prepareData(img, mask):
    mask = mask.clip(max=1)
    # segmented = blockshaped(mask.reshape(400,400),10,10)
    # for i in range(len(segmented)):
    #     if(np.max(segmented[i]) != 0):
    #         segmented[i,:,:]=1
    # segmented = unblockshaped(segmented,400,400)
    # segmented = segmented.reshape(400,400,1)
    img = img/255
    return (img,mask)

def get_data_paths(parent_dir,tag):
    input_dir = os.path.join(parent_dir,tag,'ColorChips')
    target_dir = os.path.join(parent_dir,tag,'05masks')
    input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ])
    
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png")
        ])
    return input_img_paths,target_img_paths

def full_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


# ======= Data Loaders =======

def load_data(path):
    data = pd.read_csv(path)
    # print(path + " loaded, " + len(data.index) + " records detected.")
    nchip_arr = []
    _05mask_array = []
    for im in data["Native_Chip_Name"]:
        img_PIL = load_img(IMG_READ_PATH + "NativeChips/" + im, color_mode = "grayscale")
        img_array = img_to_array(img_PIL)
        #img_array = img_array.reshape([-1,400, 400,1])
        nchip_arr.append(np.asarray(img_array)/255)

    for mask in data["05min_Mask_Name"]:
        _05mask = load_img(IMG_READ_PATH + "05masks/" + mask, color_mode="grayscale")
        _05mask_arr = img_to_array(_05mask)
        _05mask_arr = _05mask_arr.clip(max=1)
        #_05mask_arr = _05mask_arr.reshape([-1,400, 400,1])

        segmented = blockshaped(_05mask_arr.reshape(400,400),10,10)
        for i in range(len(segmented)):
            if(np.max(segmented[i]) != 0):
                segmented[i,:,:]=1
        segmented = unblockshaped(segmented,400,400)
        segmented = segmented.reshape(400,400,1)

        _05mask_array.append(np.asarray(segmented))

    return np.asarray(nchip_arr), np.asarray(_05mask_array)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (400,400),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img, mask = prepareData(img, mask)
        print(len(mask))
        yield (img,mask)



    


class DataGather(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths,shuffle=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        if shuffle==True:
            tmp = list(zip(self.input_img_paths,self.target_img_paths))
            random.shuffle(tmp)
            self.input_img_paths,self.target_img_paths = list(zip(*tmp))
            
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.array(load_img(path, target_size=self.img_size))/255.0
            x[j] = img
                
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = io.imread(path)
            img = (img>0)*1 #convert to binary
            img = img.astype('float32')           
            y[j] = np.expand_dims(img, 2)
        return x, y    

