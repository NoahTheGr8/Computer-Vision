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

def brightest_region(image,reg_rows,reg_cols):
    brightest_col = np.random.randint(0,image.shape[0]-reg_rows)
    brightest_row = np.random.randint(0,image.shape[1]-reg_cols)
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
        
        print(image.shape)

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

        fig, ax = show_image(image,'Brightest regions',save_im=False)

        reg_rows, reg_cols = 30,20
        x,y = brightest_region(image,reg_rows, reg_cols)
        ax.plot(x,y,color='k')

        x,y = brightest_region(-image,reg_rows, reg_cols)
        ax.plot(x,y,color='w')

        c = 'rgb'
        for i in range(3):
            x,y = brightest_region(color_index(image,i),reg_rows, reg_cols)
            ax.plot(x,y,color=c[i])

        fig.savefig('brightest_regions.jpg',bbox_inches='tight', pad_inches=0.1)