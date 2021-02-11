# Simple program to read images dataset

import numpy as np
import matplotlib.pyplot as plt
import utils

data_dir = 'C:\\Users\\npizz\\Desktop\\Computer Vision\\Labs\\lab2\\' 
data_files =  ['image_set_gl.npz','image_set_ag.npz','image_set_mc.npz']

plt.close('all')

for d, data_file in enumerate(data_files):
    images = np.load(data_dir+data_file)['images']
    print('array shape:', images.shape)
    rand_choice = np.sort(np.random.randint(0,images.shape[0],3))
    image_list = [images[i] for i in rand_choice]
    titles = ['images[{}]'.format(i) for i in rand_choice]
    fig_title= 'Random images from '+data_file
    utils.show_images(image_list, titles, fig_title) # Show some images

    """
    Process best possible image in each data set:
        1. Find the average image then display it
        2. Find the brightest region of the average image and box it in
        3. Select the top 10% best images by calculating the brightest regions (same regions) then sorting then choosing
        4. Align the images by aligning the brightest regions of every image by using the average as the reference
        5. Then average those top images excluding the average image
        6. Perform the 4 filters and print them
    """

    
    #----------------------- 1 -----------------------
    avg_img = np.mean(images,axis=0)
    fig,ax = utils.show_image(avg_img, title = "mean image") # Show some images
        
    ##----------------------- 2 -----------------------
    avg_br,avg_bc,avg_rs = utils.brightest_region(avg_img, 15, 15) #avg_rs is the sum of the intensities in the brightest region
    #makebox(x,y,dy,dx) | x is columns and y are rows
    xs,ys = utils.make_box(avg_bc,avg_br,15,15)
    ax.plot(xs,ys,color='r')
    
    #----------------------- 3 -----------------------
    region_sums = [] #stores stores the brightest region sums for all the images
    for image in images:
        br,bc,rs = utils.brightest_region(image, 15, 15)
        region_sums.append(rs)
    
    #grab the indices of the top 10% images | since they are sorted we need to grab the last 10% not the first
    hq_images_indices = np.argsort(region_sums)[(images.shape[0])-int(images.shape[0] *.10):]

    top_images = images[hq_images_indices]
    
    print("TOP IMAGES SHAPE > ",top_images.shape)
    
    utils.show_images(top_images[0:4], fig_title ="TOP 4 HQ IMAGES")
    
    #----------------------- 4 -----------------------
    
    
    break