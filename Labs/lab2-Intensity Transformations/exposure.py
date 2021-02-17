'''

skimage.exposure.adjust_gamma(image[, …])

Performs Gamma Correction on the input image.

skimage.exposure.adjust_log(image[, gain, inv])

Performs Logarithmic correction on the input image.

skimage.exposure.adjust_sigmoid(image[, …])

Performs Sigmoid Correction on the input image.

skimage.exposure.cumulative_distribution(image)

Return cumulative distribution function (cdf) for the given image.

skimage.exposure.equalize_adapthist(image[, …])

Contrast Limited Adaptive Histogram Equalization (CLAHE).

skimage.exposure.equalize_hist(image[, …])

Return image after histogram equalization.

skimage.exposure.histogram(image[, nbins, …])

Return histogram of image.

skimage.exposure.is_low_contrast(image[, …])

Determine if an image is low contrast.

skimage.exposure.match_histograms(image, …)

Adjust an image so that its cumulative histogram matches that of another.

skimage.exposure.rescale_intensity(image[, …])

Return image after stretching or shrinking its intensity levels.







'''
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
import matplotlib.image as mpimg

from skimage import data, img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

if __name__ == "__main__":
    plt.close("all")

    image_dir = '.\\images\\'
    image_files = os.listdir(image_dir)

    for image_file in image_files[:]:
        
        img = mpimg.imread(image_dir+image_file)

        g_img,t  = [],[]
        for g in [0.5,1,1.5]:
            g_img.append(exposure.adjust_gamma(img,gamma=g))
            t.append('gamma = '+str(g))

        utils.show_images(g_img, titles=t, fig_title='Gamma adjustment')


        g_img,t  = [],[]
        for g in [0.5,1,1.5]:
            g_img.append(exposure.adjust_log(img,gain=g))
            t.append('gain = '+str(g))

        utils.show_images(g_img, titles=t,fig_title='log adjustment')

        g_img,t  = [],[]
        for g in range(4,13,2):
            t.append('gain = '+str(g))
            g_img.append(exposure.adjust_sigmoid(img,gain=g))

        utils.show_images(g_img, titles=t, fig_title='sigmoid adjustment')
