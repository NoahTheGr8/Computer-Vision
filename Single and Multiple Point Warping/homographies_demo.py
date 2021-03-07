from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from real_index import *
from skimage import transform as tf

plt.close('all')

img = mpimg.imread('mona_lisa.jpg')
img = img/np.amax(img)



tform0 = tf.EuclideanTransform(translation=(100,100))
tform1 = tf.EuclideanTransform(rotation=15*np.pi/180, translation=(100,100))
tform2 = tf.SimilarityTransform(scale=(0.75,0.5), rotation=15*np.pi/180, translation=(100,100))
tform3 = tf.AffineTransform(shear=20*np.pi/180, scale=(0.75,0.5), rotation=15*np.pi/180, translation=(100,100))
matrix = np.array([[1, -0.3, 100],
                   [0.1, 0.9,100],
                   [0.0002, -0.0001, 1]])
tform4 = tf.ProjectiveTransform(matrix=matrix)
names = ['Translation','Euclidean','Similarity','Affine','Projective']

for i, H in enumerate([tform0, tform1,tform2,tform3,tform4]):
    t = names[i]
    print(t, 'transform')
    print('H=\n',H.params)
    canvas = np.zeros((img.shape[0]+400,img.shape[1]+400,3))
    canvas[200:-200,200:-200] = img
    warped = tf.warp(canvas, H)
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].imshow(canvas)
    ax[1].imshow(warped)
    ax[0].set_title('Original')
    ax[1].set_title(t+' transform')

for i, H in enumerate([tform0, tform1,tform2,tform3,tform4]):
    t = names[i]
    print(t, 'transform')
    print('H=\n',H.params)
    canvas = np.zeros((img.shape[0]+400,img.shape[1]+400,3))
    canvas[200:-200,200:-200] = img
    warped = tf.warp(canvas, H.inverse)
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].imshow(canvas)
    ax[1].imshow(warped)
    ax[0].set_title('Original')
    ax[1].set_title(t+' transform')
