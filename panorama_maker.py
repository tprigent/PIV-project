import sys
import os
import scipy
import numpy as np
import trans_func
from PIL import Image


def blendPanorama(im1,im2):
    blend = np.zeros((im1.shape[0], im1.shape[1], im1.shape[2])).astype(np.uint8)

    #im2.resize((im1.shape[0], im1.shape[1], im1.shape[2]),refcheck=False)
    for i, line in enumerate(blend):
        for j, pixel in enumerate(line):
            if(im2[i][j].all()==0):
                blend[i][j] = im1[i][j] 
            elif(im1[i][j].all()==0):
                blend[i][j] = im2[i][j] 
            else:
                blend[i][j] = 0.5*im1[i][j]+0.5*im2[i][j]
    return blend

def makePanorama(ref_number,input_path,output_path):
    # open template image
    template_name = input_path+'rgb_{}.jpg'.format(str(ref_number).zfill(4))
    template_frame = Image.open(template_name)
    template_frame = np.array(template_frame)

    xpixels = template_frame.shape[0]*2
    ypixels = template_frame.shape[1]*2
    #template_frame.resize((xpixels,ypixels,3),refcheck=False)


    #panorama = np.zeros((xpixels,ypixels,3)).astype(np.uint8)
    panorama = template_frame

    #panorama = blendPanorama(panorama,template_frame)

    files = os.listdir(input_path)
    num_input = int(len(files) / 2)

    start = 1
    end = num_input+1

    for iteration_counter in range(start, end):
        if(iteration_counter == ref_number):
            continue

        input_name = input_path+'rgb_{}.jpg'.format(str(iteration_counter).zfill(4))
        input_frame = Image.open(input_name)
        input_frame = np.array(input_frame)
        #input_frame.resize((xpixels,ypixels,3),refcheck=False)

        H_mat_name = 'H_{}.mat'.format(str(iteration_counter).zfill(4))
        H_mat_packed = scipy.io.loadmat(output_path+H_mat_name)
        H = np.array(H_mat_packed['H'])

        new_frame = trans_func.transform_im(H, input_frame, template_frame)

        panorama = blendPanorama(panorama,new_frame)
        new_frame_name = 'tf{}.jpg'.format(str(iteration_counter).zfill(4))
        panorama_im = Image.fromarray(panorama)
        panorama_im.save(output_path+new_frame_name)


if __name__ == "__main__":
    makePanorama(1,'data/','results/')

