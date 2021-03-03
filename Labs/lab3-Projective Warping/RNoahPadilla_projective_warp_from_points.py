import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
from utils import *

def multi_warp(img,img2):
        
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(img)
    fig.suptitle('Original image - Select 8 points', fontsize=14)
    
    #Input the source points
    print("Click 8 source points")
    srcs=[]
    srcs.append(np.asarray(plt.ginput(n=4)))
    srcs.append(np.asarray(plt.ginput(n=4)))
    
    # The destination are the four corners of the image
    dest_rows,dest_cols = 300,200
    dest = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[dest_cols,dest_rows]])
    
    for src in srcs:
        
        '''
        # Display point correspondences -points
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].imshow(img)
        ax[1].imshow(np.zeros((dest_rows,dest_cols)),cmap=plt.cm.gray)
        ax[0].set_title('Source points')
        ax[1].set_title('Destination points')
        color='rgby'
        for i in range(src.shape[0]):
            ax[0].plot(src[i, 0], src[i, 1], '*',color=color[i])
            ax[1].plot(dest[i, 0], dest[i, 1], '*',color=color[i])
        ax[1].set_xlim([-5, dest_cols+5])
        ax[1].set_ylim([dest_rows+5,-5])
        fig.suptitle('Point correspondences', fontsize=14)
        '''
        
        # Compute homography from points
        H0 = tf.ProjectiveTransform()
        H0.estimate(src, dest)
        
        # Find destination image
        warped = tf.warp(img, H0.inverse, output_shape=(np.amax(dest[:,1])+1, np.amax(dest[:,0])+1))
        
        '''
        # Display source and destination images
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[1].imshow(warped, cmap=plt.cm.gray)
        ax[0].set_title('Source image')
        ax[1].set_title('Destination image')
        fig.suptitle('Image transformation', fontsize=14)
        '''
        
        src2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])
        dest2  = src   # Destination points are the source points in the first image
        
        '''
        # Display point correspondences
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].imshow(img2)
        ax[1].imshow(img)
        color='rgby'
        for i in range(src.shape[0]):
            ax[0].plot(src2[i, 0], src2[i, 1], '*',color=color[i])
            ax[1].plot(dest2[i, 0], dest2[i, 1], '*',color=color[i])
        ax[0].set_xlim([-10, img2.shape[1]+10])
        ax[0].set_ylim([img2.shape[0]+10,-10])
        fig.suptitle('Point correspondences', fontsize=14)
        '''
        
        H1 = tf.ProjectiveTransform()
        H1.estimate(src2, dest2)
        warped = tf.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))
        
        '''
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        ax[0].imshow(img2)
        ax[1].imshow(warped)
        ax[0].set_title('Source image')
        ax[1].set_title('Destination image')
        fig.suptitle('Transformed source image', fontsize=14)
        '''
        
        # Finding mask
        # mask[r,c] = 0 if warped[r,c] = [0,0,0], otherwise mask[r,c] = 1
        mask = np.expand_dims(np.sum(warped,axis=2)==0,axis=2).astype(np.int)
        # Combining masked source and warped image
        combined = warped + img*mask
        
        #transform then transform again with the old image
        img = combined
        
        '''
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        ax[0].imshow(img*mask)
        ax[1].imshow(warped)
        ax[2].imshow(combined)
        ax[0].set_title('Original * mask')
        ax[1].set_title('Destination')
        ax[2].set_title('Original * mask + destination')
        fig.suptitle('Final image creation', fontsize=14)
        '''
        
    return img

#img is image with green screen | img2 is image that we are inserting
def green_screen(img,img2):
    
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(img)
    fig.suptitle('Original image - Select 4 points', fontsize=14)
    
    src = np.asarray(plt.ginput(n=4))
    #src = np.asarray([[19,191], [1230,184] , [1234,372], [27,387] ])
        
    # The destination are the four corners of the image
    dest_rows,dest_cols = 300,200
    dest = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[dest_cols,dest_rows]])
    
    # Display point correspondences -points
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[1].imshow(np.zeros((dest_rows,dest_cols)),cmap=plt.cm.gray)
    ax[0].set_title('Source points')
    ax[1].set_title('Destination points')
    color='rgby'
    for i in range(src.shape[0]):
        ax[0].plot(src[i, 0], src[i, 1], '*',color=color[i])
        ax[1].plot(dest[i, 0], dest[i, 1], '*',color=color[i])
    ax[1].set_xlim([-5, dest_cols+5])
    ax[1].set_ylim([dest_rows+5,-5])
    fig.suptitle('Point correspondences', fontsize=14)
            
    # Compute homography from points
    H0 = tf.ProjectiveTransform()
    H0.estimate(src, dest)
    
    # Find destination image
    warped = tf.warp(img, H0.inverse, output_shape=(np.amax(dest[:,1])+1, np.amax(dest[:,0])+1))
    
    ''' 
    # Display source and destination images
    img2 = mpimg.imread('vll_utep_720.jpg')
    img2 = img2/np.amax(img2)
    '''
    
    src2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])
    dest2  = src   # Destination points are the source points in the first image
    
    H1 = tf.ProjectiveTransform()
    H1.estimate(src2, dest2)
    warped = tf.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))
    
    '''
    Goal: Only overlay on greenscreen and not everything else
    '''   
    
    # Finding mask
    # mask[r,c] = 0 if warped[r,c] = [0,0,0], otherwise mask[r,c] = 1 // 1 is background of mask and 0 is the overlayer where image will go
    mask = np.expand_dims(np.sum(warped,axis=2)==0,axis=2).astype(np.int) #mask shape (730,1296,1)
    
    #Calculate the green screen values | a 2D array booleans at each index in the image that are valid green screen vals
    gs  = np.asarray(color_index(img,1) > .200)#gs shape = (730, 1296)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            
            #Setting the mask and the ad to only go on the green screen
            if gs[i,j] == False: 
                mask[i,j] = 1
                warped[i,j] = 0
    
    # Combining masked source and warped image
    #make sure that we overlay our add on the green screen only
    
    #(730,1296,3)
    combined = warped + img * mask
    
    #transform then transform again with the old image
    img = combined
    
    return img
    

if __name__ == "__main__":
        
    #------------------- USING REQUIRED IMAGES
    
    #1 projective warping
    img = mpimg.imread('mobile_billboard.jpg')
    img = img/np.amax(img)
    # Display source and destination images
    img2 = mpimg.imread('utep720.jpg')
    img2 = img2/np.amax(img2)
    warped_img = multi_warp(img,img2)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].imshow(img)
    ax[1].imshow(warped_img)  
    
    #3 Green screening
    img = mpimg.imread('soto.jpg')
    img = img/np.amax(img)
    # Display source and destination images
    img2 = mpimg.imread('vll_utep_720.jpg')
    img2 = img2/np.amax(img2)
    warped_img = green_screen(img,img2)
    fig, bx = plt.subplots(ncols=2, figsize=(12, 4))
    bx[0].imshow(img)
    bx[1].imshow(warped_img)
    
    #------------------- USING MY OWN IMAGES
    
    #1 projective warping
    img = mpimg.imread('timesquare.jpg')
    img = img/np.amax(img)
    # Display source and destination images
    img2 = mpimg.imread('bernie.jpg')
    img2 = img2/np.amax(img2)
    #1 projective warping
    warped_img = multi_warp(img,img2)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].imshow(img)
    ax[1].imshow(warped_img)  
    
    #3 Green screening
    img = mpimg.imread('wimbeldon.jpg')
    img = img/np.amax(img)
    # Display source and destination images
    img2 = mpimg.imread('hole.jpg')
    img2 = img2/np.amax(img2)
    warped_img = green_screen(img,img2)
    fig, bx = plt.subplots(ncols=2, figsize=(12, 4))
    bx[0].imshow(img)
    bx[1].imshow(warped_img)
        