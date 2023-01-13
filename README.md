# Automatic perspective correction

## Context
This project has been realised in the context of a computer vision course to robotics at [IST Liboa](https://tecnico.ulisboa.pt/).

The problem has been solved in a limited amount of time (around 3 weeks). Results are the ones presented for evaluation.

The global aim was, given the template (*i.e.* a scanned document) and a dataset composed of pictures of the 
document being filled by someone, to correct the perspective as if the camera was above.

One main objective was to avoid using the OpenCV library, which can solve this problem in a very few lines of code


###### insert picture

## Steps
The general algorithm involves multiple steps, which implies image processing techniques and algebra.

They can be summarized by the following points:

### 1. Extract data from the image using SIFT algorithm (OpenCV).
It gives some keypoints (points that are main elements of the images) and related descriptors (vectors describing the 
image) at these specific points. 

As it was not requested in the evaluation, the code is present in the [sift_detect.py](sift_detect.py) file.


Points were also given by the professor, and were more accurate than ours at some points (see [issues](#encountered-issues)).

### 2. Match the keypoints using descriptors
As a first approach, keypoints can be matched (between template and each picture) using the similarities between their descriptors (estimated by computing euclidian distance between the vectors).

This part of the algorithm generates both good matches and wrong matches. 
A first filter can be established regarding the distance found. If it is higher than a certain threshold, the match can be discarded.

### 3. Find accurate matches using RANSAC to determine the homography matrix
Once the keypoints pre-matched, we can use RANSAC to find the best homography matrix that will provide the best result.

The general principle is the following:
- Select 4 random matched keypoints
- Compute the homography matrix
- Apply it to each keypoint of the picture to process, and compute the euclidian distance between the new point and the corresponding one in the template.
- If the error is lower than a certain threshold, consider the corresponding keypoint as an inlier
- If the global images as a certain number of inlier, recompute the homography on all the inliers and save it as current best result
- Iterate the past steps a certain number of time, and replace the result if a better result is found

In the end, we get the best homography possible between the input image and the shape of the template, according to the input parameters.

### 4. Render the final result
In this step, the homography is simply applied to the picture to get the final result. 

As it was not requested in the evaluation, the code is present in the [render.py](render.py) file.


## Side task: panorama
A second objective was to use the previous code to generate a panorama, using overlapping pictures.
We simply had to compute the corresponding homography matrix (using a higher tolerance), and superimpose images.


## Encountered issues
During this project, we encountered few issues, with mainly:
- While using our keypoints, since the document was progressively filled out, at some point all the keypoints merged on the writings only.
After few researches, we discovered that it was related to the inner functioning of SIFT algorithm, that was focusing on the sharpest contrasts.
A possible solution to this issue was to set up better the parameters of SIFT, but this option was abandoned due to a lack of time. 
- The final output of our panorama task was not very satisfying, as the images were well superimposed but still to the size
of the input image. The matching was done correctly but we were unable to see all the data.


## Run the code
To run our code, follow the three steps:
1. ###### Generate homography matrices
```bash
python3 pivproject2022_task1.py template/templateSNS.jpg data/ output_folder/
```
2. ###### Render results
```bash
python3 render.py
```
3. ###### Generate panorama
```bash
python3 pivproject2022_task2.py nb_of_frames input_folder/ output_folder/
```

## References
- PIV course and problematics, J.P Costeira & Carlos Jorge Andrade Mariz Santiago (IST, 2022)
- OpenCV Python tutorials, [docs.opencv.org](https://docs.opencv.org/)
- OpenCV Feature Matching — SIFT Algorithm, Druga Prasad on [Medium](https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253)
- Image Stitching to create a Panorama, Naveksha Soof on [Medium](https://medium.com/@navekshasood/image-stitching-to-create-a-panorama-5e030ecc8f7)
