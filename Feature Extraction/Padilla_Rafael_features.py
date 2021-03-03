import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd
import utils
from scipy import signal
import math

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

#will take a subimage
def histogram_of_gradients(img,bins=12):
    
    '''
    USED THE FORMULA FROM THE SLIDES
    1. Find g_m by g_v and g_h
    2. Find g_d
    3. Pass image through np.hist with gd as x and gm as y values
    4. Return the HOG of the current subimage 'img'
    '''
    g_v = utils.vert_edges(img)
    g_h = utils.hor_edges(img)
    
    g_m = np.sqrt( np.power(g_v,2) + np.power(g_h,2) )

    #radian * 180/pi = degrees
    g_d = np.arctan2(g_v,g_h)

    #direction in x and magnitude on y
    hist, bin_edges = np.histogram(g_d, bins=bins, range=(-math.pi,math.pi), weights=g_m)
    
    return hist

if __name__ == "__main__":

    plt.close('all')
    image_dir = '.\\images\\'

    image_list = read_images(image_dir)

    #---------------------------2.1---------------------------
    #plot the rgb histograms next to the image for all images
    bins = 32
    for i,image in enumerate(image_list):        
        h = color_histogram_1D(image,bins)
        '''
        fig, ax = plt.subplots(ncols=4,figsize=(12, 4))
        
        ax[0].imshow(image_list[i])
        #           bins        ,      values   
        ax[1].plot(h[0][1][0:-1],h[0][0][:],color='red') #Red plot
        ax[2].plot(h[1][1][0:-1],h[1][0][:],color='green') #Green plot
        ax[3].plot(h[2][1][0:-1],h[2][0][:],color='blue') #Blue plot
        '''
    #---------------------------2.2---------------------------    
    '''
    FIND THE DESCRIPTOR FOR EACH IMAGE SUCH THAT EACH IMAGE DESCRIPTOR IS APPENDED
    TO 'add_descriptors' AND EACH DESCRIPTOR IS A 1D LIST THAT CONTAINS RGB hist outputs and HOG hist outputs
    '''
    
    #all_descriptors contains all the descriptors for each image
    all_descriptors = []
     
    for i,image in enumerate(image_list):
        
        curr_descriptor = []
        
        #all the histograms for r,g,b are stored in h
        h = color_histogram_1D(image,bins)
        
        #get the outputs (not the bin_edges) of the rgb histograms and concat the r,g,b,tl,bl,tr,br
        r_h = h[0][0]
        g_h = h[1][0]
        b_h = h[2][0]
        
        curr_descriptor.extend(r_h)
        curr_descriptor.extend(g_h)
        curr_descriptor.extend(b_h)
        
        #print(image.shape) #(620, 930, 3)#used for debugging
        
        #x = image.shape[] 0<->930
        #y = image.shape[] 0 ^ v 620
        #the 4 quadrants of the image | tl=top left ; br = bottom right
        tl = image[:image.shape[0]//2, :image.shape[1]//2] 
        bl = image[image.shape[0]//2:, :image.shape[1]//2]
        tr = image[:image.shape[0]//2, image.shape[1]//2:]
        br = image[image.shape[0]//2:, image.shape[1]//2:]
        
        '''
        #USED FOR TESTING ; plot each subimage
        fig, nx = plt.subplots(ncols=4,figsize=(12, 4))
        nx[0].imshow(tl)
        nx[1].imshow(bl)
        nx[2].imshow(tr)
        nx[3].imshow(br)
        '''
        
        subimages = []
        subimages.append(tl)
        subimages.append(bl)
        subimages.append(tr)
        subimages.append(br)
        
        for i,subimage in enumerate(subimages):
            
            hist = histogram_of_gradients(subimage)
            #USED FOR TESTING ; plot each subimage with its HOG in radians
            #fig, nx = plt.subplots(ncols=2,figsize=(12, 4))
            #nx[0].imshow(tl) #print subimage
            #nx[1].plot(hist) #plot HOG of this subimage
            curr_descriptor.extend(hist)
        
        all_descriptors.append(curr_descriptor)
        
    #---------------------------3---------------------------
    '''
    FIND MOST COMMON IMAGE
    
    #similar to closest number | descriptors are stored in same order as image_list
    '''
    
    for i in range(len(all_descriptors)):
        
        curr_desc = np.array(all_descriptors[i])
        closest_descriptor_ind = 0 #initially set to the first index
        closeset_descriptor_dist = math.inf #initially set to a really high value
        
        for j in range(len(all_descriptors)):
           
            #if we are not looking at same descriptor and its more alike than the past closest descriptor dist
            new_dist = np.linalg.norm(curr_desc - np.array(all_descriptors[j])) 
            if( (i != j) and ( new_dist < closeset_descriptor_dist )):
                
                closest_descriptor_ind = j #initially set to the first value but will update in the forloop
                closeset_descriptor_dist = new_dist
                
        #plot the current image and the closest image
        fig, nx = plt.subplots(ncols=2,figsize=(12, 4))
        nx[0].imshow(image_list[i]) #print subimage
        nx[1].imshow(image_list[closest_descriptor_ind]) #plot HOG of this subimage