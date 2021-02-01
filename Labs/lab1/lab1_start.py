"""
Author: R Noah Padilla
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

def show_image(image,title='',save_im=True,filename=None):
    # Display image in new window
    fig, ax = plt.subplots()
    ax.imshow(image,cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    if save_im:
        if filename==None:
            filename=title
        fig.savefig(filename+'.png',bbox_inches='tight', pad_inches=0.1)
    return fig, ax

def show_images(images,titles=None,save_im=True,filename=None):
    if titles==None:
        titles = ['' for i in range(len(images))]
    # Display image in new window
    fig, ax = plt.subplots(1,len(images),figsize=(12, 4))
    for i in range(len(images)):
        ax[i].imshow(images[i],cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(titles[i])
    if save_im:
        if filename==None:
            filename='show_images'
        fig.savefig(filename+'.jpg',bbox_inches='tight', pad_inches=0.1)
    return fig, ax

#return a image that is 3 channels deep however you retrieve the color_index
def color_index(image,index):
    #red
    if index==0:
        image = 2 * image[:,:,0] - image[:,:,1] - image[:,:,2]
    #green
    if index==1:
        image = 2 * image[:,:,1] - image[:,:,0] - image[:,:,2]
    #blue
    if index==2:
        image = 2 * image[:,:,2] - image[:,:,0] - image[:,:,1]
    
    return image

def get_channels(image):
    
    imageR = image[:,:,0]
    imageG = image[:,:,1]
    imageB = image[:,:,2]
    
    return [imageR, imageG, imageB]

def subsample(image,r,c):
    #print('image', image.shape)
    #print('image new ', image[::r,::c].shape)
    return image[::r,::c]

def gray_level(image):
    image = (.299*image[:,:,0]) + (.587*image[:,:,1]) + (.114*image[:,:,2])
    #print('gray level depth - ', image.shape) #results are 2D array
    return image

def negative_gray_level(image):
    image = -1 * gray_level(image)
    return image

#changes in intensity horizontally : take 1st col and sub 2nd col, 2nd col sub 3rd col, 3rd col and sub 4th col and so on.
def vert_edges(image):
    
    gray_image = gray_level(image)
    
    #since this is only messing with horizontal values (values to L and R of eachother) then we only look at columns
    image = gray_image - np.hstack( (gray_image[:,1:], np.zeros((gray_image.shape[0],1)))) #add the extra col of 0s with Horizontal stack
    
    return image

#changes in intensity vertically : take 1st row and sub 2nd row, 2nd row and sub 3rd row, 3rd row and sub 4th and so on.
def hor_edges(image):
    
   gray_image = gray_level(image)
   #since this is only messing with the vertical values then (up and down) then we only mess with rows
   image = gray_image - np.vstack( (gray_image[1:], np.zeros((1,gray_image.shape[1])))) #add the extra row of 0s with Vertical stack

   return image

def mirror(image):
    #since its mirroring we ownly mess with the columns being flipped
    return image[:,::-1,:]

def upside_down(image):
    return image[::-1]

'''
Write a function to find the brightest region of size r  c in an image, where r and c are user-provided integer
parameters. The brightest region of size rc in an image I is the subarray I[i : i+r; j : j +c] of I that has the
highest sum (or, equivalently, mean) for all valid values of i and j. For every valid value of i and j, the mean
or average intensity of the region of size r  c with top-left corner (i; j) is given by: np.mean(I[i:i+r,j:j+c]).
    1. Find the brightest region of size r  c in the intensity image and draw a rectangle surrounding it.
    2. Find the darkest region of size r  c in the intensity image and draw a rectangle surrounding it (notice
    that the darkest region is the same as the brightest region in the negative image).
    3. Find the brightest region of size r  c in the red index image and draw a rectangle surrounding it.
    4. Find the brightest region of size r  c in the green index image and draw a rectangle surrounding it.
    5. Find the brightest region of size r  c in the blue index image and draw a rectangle surrounding it.
'''
def brightest_region(image,reg_rows,reg_cols):
    
    """
    #for testing
    print("shape",image.shape)
    print("shape 0",image.shape[0])
    print("shape 1",image.shape[1])
    print("bc",brightest_col)
    print("br",brightest_row)
    """
    #-----------ORIGINAL ALGORITHM 
    #we know that the brightest colors rgb(255,255,255) is white and rgb(0,0,0) is black
    brightest_col = np.random.randint(0,image.shape[0]-reg_rows)
    brightest_row = np.random.randint(0,image.shape[1]-reg_cols)
    
    for i in range( image.shape[0] - reg_rows ):
        for j in range( image.shape[1] - reg_cols ):
         if np.sum(image[i:i+reg_rows, j:j+reg_cols]) > np.sum(image[brightest_row:brightest_row+reg_rows, brightest_col:brightest_col+reg_cols]):       
             brightest_row = i
             brightest_col = j
    
    """
    #OPTIMIZED ALGORITHM - delete one for-loop
    for i in range( image.shape[0] - reg_rows ):
        bright_region_row_i = np.argmax(np.sum(image[i:i+reg_rows,:,1]))
    """
    
    #Didnt touch this
    x = brightest_col + np.array([0,1,1,0,0])*reg_cols - 0.5
    y = brightest_row + np.array([0,0,1,1,0])*reg_rows - 0.5
    
    return x,y

if __name__ == "__main__":

    plt.close("all")
    image_dir = '.\\images_lab1\\'

    image_files = os.listdir(image_dir)

    for image_file in image_files[:]:

        image = mpimg.imread(image_dir+image_file)
        print('Image shape',image.shape)
        if np.amax(image)>1:
            image = (image/255).astype(np.float32)
            print('Converted image to float')

        show_image(image,'Original image')
        
        show_image(subsample(image,2,2),'Subsampled image')

        show_images(get_channels(image),['Red','Green','Blue'],filename='channels')

        col_ind = [color_index(image,index) for index in range(3)]
        show_images(col_ind,['Red','Green','Blue'],filename='color indices')

        show_image(gray_level(image),'Gray-level image')

        show_image(negative_gray_level(image),'Negative gray-level image')

        show_image(mirror(image),'Mirrored image')

        show_image(upside_down(image),'Upside-down image')

        show_image(vert_edges(image),'Vertical edges') #gray level

        show_image(hor_edges(image),'Horizontal edges') #gray level

        show_image(np.sqrt(hor_edges(image)**2+vert_edges(image)**2),'Edge magnitudes')


        #---------------------------------------------------------------------------------
        fig, ax = show_image(image,'Brightest regions',save_im=False)

        reg_rows, reg_cols = 30,40
        #
        x,y = brightest_region(gray_level(image),reg_rows, reg_cols)
        ax.plot(x,y,color='k')#k = black
        
        #Darkest color
        x,y = brightest_region(-gray_level(image),reg_rows, reg_cols)
        ax.plot(x,y,color='w')#w = white

        c = 'rgb'
        for i in range(3):
            #passing each layer
            x,y = brightest_region(color_index(image,i),reg_rows, reg_cols)
            ax.plot(x,y,color=c[i])

        fig.savefig('brightest_regions.jpg',bbox_inches='tight', pad_inches=0.1)
        