from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg


def show_image(image,title='',save_im=False,filename=None):
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

def sub_image(I,r0,c0,rows,cols):
    return I[r0:r0+rows,c0:c0+cols]

#expects a 2D array so make it work for 3D
def correlate2d(image,filt):
    r,c = filt.shape
    result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1))
    
    r,c = filt.shape
    result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1))
    
    r,c = filt.shape
    result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1))
    
    #extra for loop?
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.sum(image[i:i+r,j:j+c]*filt) # Dot product of image region and filter
    
    return result

def gaussian_filter(size, sigma):
    d = ((np.arange(size) - (size-1)/2)**2).reshape(-1,1)
    f = np.exp(-(d+d.T)/2/sigma/sigma)
    return f/np.sum(f)

def correlate2d_scipy(image,filt):
    return signal.correlate2d(image, filt,mode='valid')

if __name__ == "__main__":
    plt.close("all")

    image = mpimg.imread('eiffel.jpg')
    image = np.mean(image,axis=2)

    f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    f2 = f1.T
    f3 = np.ones((3,3))/9
    f4 = np.zeros((3,3))
    f4[1,1]=1
    f5 = 2*f4-f3
    f6 = gaussian_filter(15, 3)
    titles=['Filter','Result','Close-up']
    k=1
    for f in [f1,f2,f3,f4,f5,f6]:
        print(f)
        print(np.sum(f))
        image_f = correlate2d_scipy(image,f)
        show_image(image_f)
        show_image(sub_image(image_f,62,325,50,25))
        k+=1
