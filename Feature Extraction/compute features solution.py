import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd
import utils
from scipy import signal
import time

def color_histogram_1D(im,bins=32):
    h = []
    for i in range(3):
        h.append(np.histogram(im[:,:,i], bins=bins,range =(0,1))[0])
    ha =np.array(h).reshape(-1)
    return ha/im.shape[0]/im.shape[1]

def color_histogram_3D(im,bins=6):
    count = np.zeros((bins,bins,bins),dtype=int)
    im_b = (im*(bins-1e-5)).astype(int)
    for i in range(bins):
        r = im_b[:,:,0] == i
        for j in range(bins):
            g = im_b[:,:,1] == j
            rg = r*g
            for k in range(bins):
                b = im_b[:,:,2] == k
                rgb = rg*b
                count[i,j,k] = np.sum(rgb)
    return count.reshape(-1)/im.shape[0]/im.shape[1]

def histogram_of_gradients(img,bins=12):
    hog = []
    gray_conv = np.array([0.2989,0.5870,0.1140]).reshape(1,1,3)
    im = np.sum(img*gray_conv,axis=2)
    f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gh = signal.correlate2d(im, f1, mode='same')
    gv = signal.correlate2d(im, f1.T, mode='same')
    g_mag = np.sqrt(gv**2+gh**2)
    g_dir = np.arctan2(gv,gh)
    bin_size = 2*np.pi/bins
    g_dir_bin = ((g_dir+np.pi)/bin_size).astype(np.int)
    for i in range(bins):
        hog.append(np.sum(g_mag[g_dir_bin==i]))
    return np.array(hog)/im.shape[0]/im.shape[1]

def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = mpimg.imread(image_dir+im)[:,:,:3]
        img=img/np.amax(img)
        image_list.append(img)
    return image_list

def compute_feature_vector(img):
    color_hist = color_histogram_1D(img)
    #color_hist = color_histogram_3D(img)
    hog00 = histogram_of_gradients(img[:img.shape[0]//2,:img.shape[1]//2])
    hog01 = histogram_of_gradients(img[:img.shape[0]//2,img.shape[1]//2:])
    hog10 = histogram_of_gradients(img[img.shape[0]//2:,:img.shape[1]//2])
    hog11 = histogram_of_gradients(img[img.shape[0]//2:,img.shape[1]//2:])
    hogs = np.hstack((hog00,hog01,hog10,hog11))
    return np.hstack((color_hist,hogs))

def get_diff_matrix(feature_list):
    diff = np.zeros((len(feature_list),len(feature_list)))
    for i in range(len(image_list)):
        for j in range(i):
            d = np.sqrt(np.sum((feature_list[i] - feature_list[j])**2))
            diff[i,j] =d
    diff = diff + diff.T
    return diff

if __name__ == "__main__":

    plt.close('all')
    image_dir = 'C:\\Users\\npizz\\Desktop\\Computer Vision\\Feature Extraction\\images\\'

    image_list = read_images(image_dir)

    feature_list = []
    for img in image_list:
        feature_list.append(compute_feature_vector(img))

    features = np.array(feature_list)

    #Normalize assuming last 48 features are gradients
    features[:,:-48] = features[:,:-48] - np.amin(features[:,:-48])
    features[:,:-48] = features[:,:-48]/np.amax(features[:,:-48])
    features[:,-48:] = features[:,-48:] - np.amin(features[:,-48:])
    features[:,-48:] = 2*features[:,-48:]/np.amax(features[:,-48:]) # Assign gradients twice the weight of colors

    diff = get_diff_matrix(features)
    match = np.argmin(diff+np.identity(diff.shape[0])*np.amax(diff),axis=1)

    for i in range(len(image_list)):
        fig, ax = plt.subplots(ncols=3,figsize=(15, 4))
        ax[0].imshow(image_list[i])
        ax[1].imshow(image_list[match[i]])
        ax[2].plot(features[i])
        ax[2].plot(features[match[i]])
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].set_title('Original image')
        ax[1].set_title('Best match')

