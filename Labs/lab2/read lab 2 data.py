# Simple program to read images dataset

import numpy as np
import matplotlib.pyplot as plt
import utils

data_dir = 'C:\\Users\\OFuentes\\OneDrive - University of Texas at El Paso\\CS4363\\labs\\data lab 2\\' # Use your own path here
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

