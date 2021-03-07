import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
from matplotlib import transforms
import cv2

def display_control_lines(im0,im1,pts0,pts1,clr_str = 'rgbycmwk'):
    canvas_shape = (max(im0.shape[0],im1.shape[0]),im0.shape[1]+im1.shape[1],3)
    canvas = np.zeros(canvas_shape,dtype=type(im0[0,0,0]))
    canvas[:im0.shape[0],:im0.shape[1]] = im0
    canvas[:im1.shape[0],im0.shape[1]:canvas.shape[1]]= im1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(canvas)
    ax.axis('off')
    pts2 = pts1+np.array([im0.shape[1],0])
    for i in range(pts0.shape[0]):
        ax.plot([pts0[i,0],pts2[i,0]],[pts0[i,1],pts2[i,1]],color=clr_str[i%len(clr_str)],linewidth=1.0)
    fig.suptitle('Point correpondences', fontsize=16)

def cond_num_and_det(H):
    print(H)
    w,v = np.linalg.eig(np.array(H.params))
    w = np.sort(np.abs(w))
    cn = w[2]/w[0]
    d = np.linalg.det(H.params)
    print('condition number {:7.3f}, determinant {:7.3f}'.format(cn,d))

def select_matches_ransac(pts0, pts1):
    
    threshhold = 10.0
    H, mask = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.RANSAC,threshhold)
    choice = np.where(mask.reshape(-1) ==1)[0]
    return pts0[choice], pts1[choice]



if __name__ == "__main__":
    # An ORB feature consists of a keypoint (the coordinates of the region's center) and a descriptor (a binary vector of length 256 that characterizes the region)
    plt.close('all')

    img1 = mpimg.imread('.//images//utepccsA.jpg')
    #img2 = mpimg.imread('.//images//utepccsB.jpg') #160 matches
    img2 = mpimg.imread('.//images//arc1.jpg')
    
    orb = cv2.ORB_create()
    
    #------------------ 1. Find and display the ORB descriptors of two input images.
    # An ORB feature consists of a keypoint (the coordinates of the region's center) and a descriptor (a binary vector of length 256 that characterizes the region)
    fig, ax = plt.subplots(ncols=2)
    keypoints1, descriptors1 = orb.detectAndCompute(img1,mask=None)
    ax[0].imshow(cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0), flags=0))
    
    keypoints2, descriptors2 = orb.detectAndCompute(img2,mask=None)
    ax[1].imshow(cv2.drawKeypoints(img2, keypoints2, None, color=(0,255,0), flags=0))
    
     #256 bits describe a region
    #hamming dist between 2 binary vectors 
    #dir(var) -> shows all attributes for var
    #print(dir(keypoints1[0])) #shows all attrib of the keypoint at index0
    #print(keypoints1[0].pt) #shows the coordinate are keypoint0
    
    #------------------ 2. Use the Brute Force matcher to find the descriptor matches.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1,descriptors2)
    
    # Extract data from orb objects and matcher
    dist = np.array([m.distance for m in matches])
    ind1 = np.array([m.queryIdx for m in matches])
    ind2 = np.array([m.trainIdx for m in matches])
    keypoints1 = np.array([p.pt for p in keypoints1])
    keypoints2 = np.array([p.pt for p in keypoints2])
    keypoints1 = keypoints1[ind1]
    keypoints2 = keypoints2[ind2]
    
    #------------------ 3. Display the first 20 descriptor matches and determine, visually, how many of them are correct
    
    sel = 20
    
    #display_control_lines(img1,img2, keypoints1[:sel], keypoints2[:sel])#connects keypoints from both images and prints those connections BUT not accurate at all
    
    #------------------ 4. Repeat the previous question, but now sort the matches by distance prior to displaying
    ds = np.argsort(dist)
    #display_control_lines(img1,img2, keypoints1[ds[:sel]], keypoints2[ds[:sel]]) #kind of accurate but could be better BASED var 'dist'
    
    #------------------ 5. Compute a homography from the matches from questions 3. Display the condition number and determinant of the homography.
   
    #To check if the points correspond to eachother multiply the points1 by H and see if there correspond
    pts0, pts1 = select_matches_ransac(keypoints1[:sel],keypoints2[:sel])
    
    display_control_lines(img1,img2,pts0, pts1)
    
    
    






















