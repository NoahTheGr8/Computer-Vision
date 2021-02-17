import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from real_index import *

plt.close('all')

#img = mpimg.imread('utep_axe_02.jpg')
img = mpimg.imread('mona_lisa.jpg')

img = img/np.amax(img) #divide by the max value to convert it to floating point

plt.figure()
plt.imshow(img)

print(img.shape)

print(">>>>>>>>>>>>> Click source and destination points <<<<<<<<<<<<<<")
p = np.asarray(plt.ginput(n=2), dtype=np.int)  #p stores reference points described in the slides
#python do while loop
while(p[0,0]!=p[1,0]) and (p[0,1]!=p[1,0]):
    
    plt.figure()
    plt.imshow(img)
    plt.plot(p[:,0],p[:,1],marker = '*',color='y')
    plt.pause(0.01)
    plt.show()
    
    cols, rows = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))  # rows, cols correspond to Y and X matrices in pseudocode
    
    #distance formula from math
    dist_r = (rows - p[1,1])**2 #rows - row index of dest point
    dist_c = (cols - p[1,0])**2 #cols - col index of dest point
    dist = np.sqrt(dist_r + dist_c)
    
    w = np.exp(-dist/100)
    
    # Replace random generation of delta_rows, delta_cols by displacement based on formula, as described in the sildes
    # delta_rows, delta_cols correspond to Dy and Dx matrices in pseudocode
    delta_rows = w * ( p[0,1] - p[1,1]) #w * (x0-x1)
    delta_cols = w * ( p[0,0] - p[1,0]) #w * (y0-y1)
    
    rows_t = rows + delta_rows
    cols_t = cols + delta_cols
    
    # We may need to clip values to valid range
    rows_t = np.clip(rows_t,0,img.shape[0]-1)
    cols_t = np.clip(cols_t,0,img.shape[1]-1)
    
    T = real_index(img, rows_t, cols_t)
    img = T
    plt.figure()
    plt.imshow(T)
    
    print(">>>>>>>>>>>>> Click source and destination points <<<<<<<<<<<<<<")
    p = np.asarray(plt.ginput(n=2), dtype=np.int)  #p stores reference points described in the slides    