import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd

def array_to_excel(A,filename):
    df = pd.DataFrame()
    for c in range(A.shape[1]):
        df[c] = A[:,c]
    df.to_excel(filename+".xlsx",header=False, index=False)

def integral_image(image):
    if len(image.shape)>2:
        image = gray_level(image)
    if 'int' in str(type(image[0,0])):
        integral_image = np.zeros((image.shape[0]+1,image.shape[1]+1),dtype=np.int32)
    else:
        integral_image = np.zeros((image.shape[0]+1,image.shape[1]+1),dtype=np.float32)
    integral_image[1:,1:] = np.cumsum(np.cumsum(image,axis=1))
    #integral_image = np.cumsum(integral_image,axis=0)
    return integral_image

def region_sums(image,reg_rows,reg_cols):
    int_im = integral_image(image)
    return int_im[reg_rows:,reg_cols:] - int_im[reg_rows:,:-reg_cols] - int_im[:-reg_rows,reg_cols:] + int_im[:-reg_rows,:-reg_cols]

def show_image(image,title='',save_im=True,filename=None):
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

def show_images(images,titles=None,fig_title='', save_im=True,filename=None):
    if titles==None:
        titles = ['' for i in range(len(images))]
    # Display image in new window
    fig, ax = plt.subplots(1,len(images),figsize=(12, 4))
    for i in range(len(images)):
        ax[i].imshow(images[i],cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(titles[i])
    fig.suptitle(fig_title, fontsize=16)
    if save_im:
        if filename==None:
            filename='show_images'
        fig.savefig(filename+'.jpg',bbox_inches='tight', pad_inches=0.1)

    return fig, ax

def color_index(image,index):
    return 3*image[:,:,index] - np.sum(image,axis=2)

def subsample(image,r,c):
    return image[::r,::c]

def gray_level(image):
    gray_conv = np.array([0.2989,0.5870,0.1140]).reshape(1,1,3)
    return np.sum(image*gray_conv,axis=2)

def negative_gray_level(image):
    return 1 - gray_level(image)

def vert_edges(gray_image):
    if len(gray_image.shape)>2:
        gray_image = gray_level(gray_image)
    edges =  np.zeros_like(gray_image)
    edges[:,:-1] = gray_image[:,:-1] -  gray_image[:,1:]
    return edges

def hor_edges(gray_image):
    if len(gray_image.shape)>2:
        gray_image = gray_level(gray_image)
    edges =  np.zeros_like(gray_image)
    edges[:-1] = gray_image[:-1] -  gray_image[1:]
    return edges

def mirror(image):
    return image[:,::-1]

def upside_down(image):
    return image[::-1]

def brightest_region(image,reg_rows,reg_cols):
    #rs = region_sums_ii(image,reg_rows,reg_cols)
    rs = region_sums(image,reg_rows,reg_cols)
    brightest = np.argmax(rs)
    brightest_col = brightest%rs.shape[1]
    brightest_row = brightest//rs.shape[1]
    x = brightest_col + np.array([0,1,1,0,0])*reg_cols - 0.5
    y = brightest_row + np.array([0,0,1,1,0])*reg_rows - 0.5
    return x,y

