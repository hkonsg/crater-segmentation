
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import helpers as h
import os,sys
from os import *
import skimage.io as io


#method to load testing images
def testGenerator(test_path,num_image = 30,target_size = (256,256)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"test_img_%d.png"%i))
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
#method to save predicted images
def save_result(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
