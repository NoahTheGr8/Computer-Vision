import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd
import utils
from scipy import signal


from skimage import data, exposure, img_as_float

def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = mpimg.imread(image_dir+im)[:,:,:3]
        img=img/np.amax(img)
        image_list.append(img)
    return image_list

def color_histogram_1D(im,bins=32):
    h = []
    for channel in range(im.shape[2]):
        
        #returns hist values, num of bins+1
        h.append(np.histogram(im[:,:,channel], bins=bins))
        
    return h

def histogram_of_gradients(img,bins=12):
    hog = np.zeros(bins)
    return hog

if __name__ == "__main__":

    plt.close('all')
    image_dir = '.\\images\\'

    image_list = read_images(image_dir)

    '''
    for i in range(len(image_list)):
        fig, ax = plt.subplots(ncols=4,figsize=(12, 4))
        ax[0].imshow(image_list[i])
        ax[1].imshow(image_list[match[i]]) #
    '''
    
    #plot the histograms next to the image
    bins = 32
    for i,image in enumerate(image_list):        
        h = color_histogram_1D(image,bins)
        fig, ax = plt.subplots(ncols=4,figsize=(12, 4))
        ax[0].imshow(image_list[i])
        #           bins        ,      values   
        ax[1].plot(h[0][1][0:-1],h[0][0][:],color='red') #Red plot
        ax[2].plot(h[1][1][0:-1],h[1][0][:],color='green') #Red plot
        ax[3].plot(h[2][1][0:-1],h[2][0][:],color='blue') #Red plot
          
