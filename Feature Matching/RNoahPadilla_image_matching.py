import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
from matplotlib import transforms
import cv2
import os

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
    # Very large condition numbers usually indicate a bad homography
    # Negative determinants and those with low absolute values usually indicate a bad homography
    # Large determinant also usually indicate a bad homography
    w,v = np.linalg.eig(np.array(H.params))
    w = np.sort(np.abs(w))
    cn = w[2]/w[0]
    d = np.linalg.det(H.params)
    return cn, d
    print('condition number {:7.3f}, determinant {:7.3f}'.format(cn,d))

def homography_error(H,pts1,pts2):
    pts1 =  np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2 =  np.hstack((pts2,np.ones((pts2.shape[0],1))))
    proj = np.matmul(H,pts1.T).T
    proj = proj/proj[:,2].reshape(-1,1)
    err = proj[:,:2] - pts2[:,:2]
    return np.mean(np.sqrt(np.sum(err**2,axis=1)))

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.RANSAC,5.0)
    choice = np.where(mask.reshape(-1) ==1)[0]
    return pts0[choice], pts1[choice]

if __name__ == "__main__":
    
    '''
    Find the most similar images for each image in the set using the RANSAC algorithm
    '''
    plt.close('all')
    # An ORB feature consists of a keypoint (the coordinates of the region's center) and a descriptor (a binary vector of length 256 that characterizes the region)
    #plt.close('all')
    
    image_dir = '.\\images\\'
    image_files = os.listdir(image_dir)
    
    '''
    1. Make 2 lists - 1 for keypoints and 1 for descriptors
    '''
    
    for i,image_file in enumerate(image_files):

        img1 = mpimg.imread(image_dir+image_file)
        
        orb = cv2.ORB_create()
    
        
        keypoints1, descriptors1 = orb.detectAndCompute(img1,mask=None)
        
        
        most_matches = 0 #num of most matches between 2 images
        most_matches_img_index = 0 #the index of the image that is the closest to image 'image_file'
        
        for j,image_file2 in enumerate(image_files):
            
            #only consider images that are not the same
            if i != j:
                
                img2 = mpimg.imread(image_dir+image_file2)
                keypoints2, descriptors2 = orb.detectAndCompute(img2,mask=None)
            
                # Create BFMatcher object
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(descriptors1,descriptors2)
                
                #find the subest of these matches that can be explained using the ransac | for every image find the best matching image
            
                # Extract data from orb objects and matcher
                dist = np.array([m.distance for m in matches])
                ind1 = np.array([m.queryIdx for m in matches])
                ind2 = np.array([m.trainIdx for m in matches])
                
                #TWEAKED BY NOAH - NEEDED TO CHANGE IF GOING THROUGH A LOOP
                k1 = np.array([p.pt for p in keypoints1])
                k2 = np.array([p.pt for p in keypoints2])
                k1 = k1[ind1]
                k2 = k2[ind2]
                # keypoints1[i] and keypoints2[i] are a match
            
                #------------------ Compute a homography from the matches from before sorting dst and display the condition number and determinant of the homography.
                
                '''
                print('Original number of matches',keypoints1.shape[0])
                H = tf.ProjectiveTransform()
                H.estimate(keypoints1, keypoints2)
                print('Resulting homography\n',H.params)
                cn,d = cond_num_and_det(H)
                print('condition number: {:7.3f}, determinant: {:7.3f}'.format(cn,d))
                print('Mean projection error {:7.3f}'.format(homography_error(H.params,keypoints1, keypoints2)))
                '''
                
                #---------------------BEST MATCH POINTS FROM BOTH IMAGES
                p0, p1 = select_matches_ransac( k1, k2 )
                
                '''
                print('Number of matches after performing RANSAC', keypoints1.shape[0])
                H.estimate(keypoints1, keypoints2)
                print('Resulting homography\n',H.params)
                cn,d = cond_num_and_det(H)
                print('condition number: {:7.3f}, determinant: {:7.3f}'.format(cn,d))
            
                #display_control_lines(img1,img2, keypoints1, keypoints2)
            
                print('Mean projection error {:7.3f}'.format( homography_error(H.params,keypoints1, keypoints2) ))
                '''
                
                '''
                Another possible approach -
                '''
                #If the current amount of matches between image_file and image_file2 is more than the previous amount of most matches then save info for this image
                if (p0.shape[0] > most_matches):
                    most_matches = p0.shape[0]
                    most_matches_img_index = j
                    
        matched_image = mpimg.imread(image_dir + image_files[most_matches_img_index])
        
        fig, ax = plt.subplots(ncols=2)
        fig.suptitle('Match ' + str(i), fontsize=16)
        ax[0].imshow(img1)
        ax[1].imshow(matched_image)