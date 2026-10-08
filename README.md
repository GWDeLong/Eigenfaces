Note: See the file [Final](Final.ipynb) for a full write up with output.

# Eigenfaces Project

## Authors
Livingston, K., DeLong, G., with special thanks to Dr. Vu H. at the University of Texas at San Antonio, and Dr. Brunton, S. at the University of Washington

## Introduction.
This project performs Principle Component Analysis (PCA) via Singular Value Decomposition (SVD) on a large image database to achieve

- Image Reconstruction
- Image Classification
- Image Matching

## Image Reconstruction
Through the use of PCA we are able to separate into images intro their key components. With said components we are able to reconstruct any image (given a large enough sample size).

## Image Classification
By projecting images into a subspace of the eigenspace created by PCA we are able to observe clear clustering of image sets and are able to classify new images.

## Image Matching
By calculating differences of images we can find the most similar images to an input image and therefore return the closest matching set of images.

## Data Set
Below is a link to the AT&T data set of faces used in this project.
https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
