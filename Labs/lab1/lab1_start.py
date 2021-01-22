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
    return image

def get_channels(image):
    return [image, image, image]

def subsample(image,r,c):
    return image

def gray_level(image):
    return image

def negative_gray_level(image):
    return image

def vert_edges(gray_image):
    return image

def hor_edges(gray_image):
   return image

def mirror(image):
    return image

def upside_down(image):
    return image

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

        show_image(subsample(image,2,2),'Subsampled image')

        show_images(get_channels(image),['Red','Green','Blue'],filename='channels')

        col_ind = [color_index(image,index) for index in range(3)]
        show_images(col_ind,['Red','Green','Blue'],filename='color indices')

        show_image(gray_level(image),'Gray-level image')

        show_image(negative_gray_level(image),'Negative gray-level image')

        show_image(mirror(image),'Mirrored image')

        show_image(upside_down(image),'Upside-down image')

        show_image(vert_edges(image),'Vertical edges')

        show_image(hor_edges(image),'Horizontal edges')

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