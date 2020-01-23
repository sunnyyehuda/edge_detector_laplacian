# objects detector infrastructure based on classic method (sharpening + laplacian + thresholding)
The program load an image and manipulate the output.
This program not classify the object, just markers its edges.

# libraries
- cv2
- numpy
- skimage.exposure

# example
There is a car image from google search just for the test, you cn change the image.

# parameters
- marker_width is the width of the points markering the edges
- threshold is the intensity of the pixel after applying convolution with the laplacian kernel

# FYI
You can change the sharpening kernel and the laplacian kernel to whatever you want.
Suggestion:
- sobel
- canny (output the edges)
