import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf

plt.close('all')

img = mpimg.imread('denver.jpg')

img = img/np.amax(img)


fig, ax = plt.subplots(figsize=(12,10))
ax.imshow(img)
fig.suptitle('Original image', fontsize=14)

# The destination are the four corners of the image
dest_rows,dest_cols = 300,200
dest = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[dest_cols,dest_rows]])

# Input the source points
print("Click four source points")
src = np.asarray(plt.ginput(n=4))

# Display point correspondences
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

# Display source and destination images
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].imshow(img, cmap=plt.cm.gray)
ax[1].imshow(warped, cmap=plt.cm.gray)
ax[0].set_title('Source image')
ax[1].set_title('Destination image')
fig.suptitle('Image transformation', fontsize=14)

# Display source and destination images
img2 = mpimg.imread('mona_lisa.jpg')
img2 = img2/np.amax(img2)

src2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])
dest2  = src   # Destination points are the source points in the first image

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

H1 = tf.ProjectiveTransform()
H1.estimate(src2, dest2)
warped = tf.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))

fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].imshow(img2)
ax[1].imshow(warped)
ax[0].set_title('Source image')
ax[1].set_title('Destination image')
fig.suptitle('Transformed source image', fontsize=14)

# Finding mask
# mask[r,c] = 0 if warped[r,c] = [0,0,0], otherwise mask[r,c] = 1
mask = np.expand_dims(np.sum(warped,axis=2)==0,axis=2).astype(np.int)
# Combining masked source and warped image
combined = warped + img*mask

fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
ax[0].imshow(img*mask)
ax[1].imshow(warped)
ax[2].imshow(combined)
ax[0].set_title('Original * mask')
ax[1].set_title('Destination')
ax[2].set_title('Original * mask + destination')
fig.suptitle('Final image creation', fontsize=14)

