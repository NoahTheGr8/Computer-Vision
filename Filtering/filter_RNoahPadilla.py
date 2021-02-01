"""
Author: Dr. Olac Fuentes

Modified by: R Noah Padilla

Modification Goals:
    
    1. Change correlate2d(image,filt) to work with both gray-level and color images. 
        The current function expect image and filt to be 2D arrays. Modify the function 
        to apply filt separately to every channel of image if image is a color image. If
        the function should return an array with the same number of dimensions as image.
        
    2. Repeat the previous question but now use the correlate2d function from scipy.signal.
    
    3. Extend edge finding to color images. The function should apply the filter 
        separately to every channel and then return a 2D array containing the 
        magnitudes of the changes in the image (that is, each pixel i,j should 
        contain the norm of the intensity changes at position i,j, in the direction 
        given by the filter (vertical or horizontal)).
        
    4. Modify the image sharpening for color images described in the slides by 
        replacing the mean filter by a gaussian filter.
        
        
Pre-requisite files: Lab1_start.py -> vertical and horizontal functions for #3
    
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

def show_image(image,title='',save_im=False,filename=None):
    # Display image in new window
    fig, ax = plt.subplots()
    ax.imshow(image,cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    if save_im:
        if filename==None:
            filename=title
        fig.savefig(filename+'.jpg',bbox_inches='tight', pad_inches=0.1)
    return fig, ax

def sub_image(I,r0,c0,rows,cols):
    return I[r0:r0+rows,c0:c0+cols]

#DIY version of 'correlate2d_scipy' that also works for 3D images
def correlate2d3d(image,filt):
    
    #print(">>> shape:",image.shape)
    #print(">>> shape size:",len(image.shape))
    
    #for non color images
    if len(image.shape) == 2:
        print(">>> Filtering 2D image... <<<")
        r,c = filt.shape
        result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i,j] = np.sum(image[i:i+r,j:j+c]*filt) # Dot product of image region and filter
                
    #for color images
    if len(image.shape) == 3:
        print(">>> Filtering 3D image... <<<")
        r,c = filt.shape
        result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1,3))
        #go through each channel
        for k in range(result.shape[2]):
            #Go through the regions in each channel
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    result[i,j,k] = np.sum(image[i:i+r,j:j+c,k]*filt) # Dot product of image region and filter and its respective channel/layer
    return result

def gaussian_filter(size, sigma):
    d = ((np.arange(size) - (size-1)/2)**2).reshape(-1,1)
    f = np.exp(-(d+d.T)/2/sigma/sigma)
    return f/np.sum(f)

def correlate2d_scipy(image,filt):
    return signal.correlate2d(image, filt,mode='valid')

if __name__ == "__main__":
    plt.close("all")

    image = mpimg.imread('eiffel.jpg')
    
    if np.amax(image) > 1:
     image = (image / 255).astype(np.float32) #normalize
     print('Converted image to float')
     
    
    #below are all the filters
    f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #vertical edge filter
    f2 = f1.T #horizontal edge filter
    f3 = np.ones((3,3))/9 #blurrier than original - gets average of neighbors
    f4 = np.zeros((3,3)) #sharper than original 
    f4[1,1]=1
    f5 = 2*f4-f3
    f6 = gaussian_filter(15, 3)
    
    #---------------------------------------  #1  ---------------------------------------
    k=1
    for f in [f1,f2,f3,f4,f5,f6]:
        print(f)
        print(np.sum(f))
        #image_f = correlate2d_scipy(image,f)
        
        #the one we did
        image_f = correlate2d3d(image,f)
        
        #shows filtered image in normal view and zoomed in
        show_image(image_f, "#1")
        show_image(sub_image(image_f,62,325,50,25), "#1")
        k+=1
        
    #---------------------------------------  #2  ---------------------------------------
    image = np.mean(image,axis=2)#THIS IS THE MEAN FILTER
    k=1
    for f in [f1,f2,f3,f4,f5,f6]:
        print(f)
        print(np.sum(f))
        
        #the one scipy does 
        image_f = correlate2d_scipy(image,f)
        
        #shows filtered image in normal view and zoomed in
        show_image(image_f, "#2")
        show_image(sub_image(image_f,62,325,50,25), "#2")
        k+=1
     
    #---------------------------------------  #3  ----------------------------------------
    image = mpimg.imread('eiffel.jpg')
    
    if np.amax(image) > 1:
     image = (image / 255).astype(np.float32) #normalize
     print('Converted image to float again')
    
    #Edge finding are filters 1 and 2 | vertical and horizontal
    k=1
    for f in [f1,f2]:
        print(f)
        print(np.sum(f))
        
        #apply filter seperately to every channel
        image_f = correlate2d3d(image,f)
        
        #Return 2D array containing the magnitudes of the changes in the image (pixel contains the norm in direction given by filter)
        I_mag = np.linalg.norm(image_f,axis=2)
            
        #shows filtered image in normal view and zoomed in
        show_image(I_mag, "#3")
        show_image(sub_image(I_mag,62,325,50,25), "#3")
        k+=1
    
    #---------------------------------------  #4  ---------------------------------------

    #replacing mean filter by gauss filter - >> DOES NOT MENTION HOW SHARP IMAGE SHOULD BE <<
    f3 = gaussian_filter(15, 1e-10)
    
    #DIY one
    image_f = correlate2d3d(image, f3)
    
    #shows filtered image in normal view and zoomed in
    show_image(image_f, "#4 Gaussian Filter")
    show_image(sub_image(image_f,62,325,50,25), "#4 Gaussian Filter")