import sys
import os
import scipy
import cv2 as cv2
import numpy as np
from tqdm import tqdm
import trans_func

def blendPanorama(im1,im2):
    blend = np.zeros((im1.shape[0], im1.shape[1], im1.shape[2])).astype(np.uint8)

    #im2.resize((im1.shape[0], im1.shape[1], im1.shape[2]),refcheck=False)
    for i, line in tqdm(enumerate(blend)):
        for j, pixel in enumerate(line):
            if(im2[i][j].all()==0):
                blend[i][j] = im1[i][j] 
            elif(im1[i][j].all()==0):
                blend[i][j] = im2[i][j] 
            else:
                blend[i][j] = 0.5*im1[i][j]+0.5*im2[i][j]
    return blend

if __name__ == "__main__":

    # open template image
    
    template_frame = cv2.imread('data/rgb_0001.jpg')
    xpixels = template_frame.shape[0]*2
    ypixels = template_frame.shape[1]*2
    #template_frame.resize((xpixels,ypixels,3),refcheck=False)


    #panorama = np.zeros((xpixels,ypixels,3)).astype(np.uint8)
    panorama = template_frame

    #panorama = blendPanorama(panorama,template_frame)

    files = os.listdir('data')
    num_input = int(len(files) / 2)

    start = 2
    end = num_input+1

    for iteration_counter in tqdm(range(start, end)):
        input_name = 'data/rgb_{}.jpg'.format(str(iteration_counter).zfill(4))
        input_frame = cv2.imread(input_name)
        #input_frame.resize((xpixels,ypixels,3),refcheck=False)

        H_mat_name = 'H_{}.mat'.format(str(iteration_counter).zfill(4))
        H_mat_packed = scipy.io.loadmat('results/'+H_mat_name)
        H = np.array(H_mat_packed['H'])

        new_frame = trans_func.transform_im(H, input_frame, template_frame)

        panorama = blendPanorama(panorama,new_frame)
        new_frame_name = 'tf{}.jpg'.format(str(iteration_counter).zfill(4))
        cv2.imwrite('results/'+new_frame_name, panorama)

