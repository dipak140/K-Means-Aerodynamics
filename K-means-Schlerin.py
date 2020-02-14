# -*- coding: utf-8 -*-
"""
@author: DIPAK
K Means and Image Segmentation for Schlieren
"""
import numpy as np
import cv2

def imageKmean(original_image, K):
    original_image2 = original_image.reshape((-1,3))  ## Reshape the image in MX3 form
    vectorized = original_image2
    vectorized = np.float32(vectorized)   ## convert type into float 32 for openCV.Kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ## 10 = number of iterations,  1.0 = epsilon
    attempts = 10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((original_image.shape))
    cv2.imshow('with K=%d.jpg'%(K,),res2)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()    

def combineImages(path):
    import sys
    from PIL import Image
    images = [Image.open(path) for x in ['with K=2.jpg', 'with K=4.jpg', 'with K=6.jpg', 'with K=2.jpg']]
    
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
        
        new_im.save('test.jpg')
        
        
if __name__ == "__main__":
    image_name = input("Enter image name: ")
    original_image = cv2.imread(image_name)
    for K in range(2,10,2):    
        imageKmean(original_image, K)
    cv2.imshow('original_image',original_image)
    path = r"C:\Users\DIPAK\COMP167\K-Means-Aerodynamics\image1"
    #combineImages(path)


