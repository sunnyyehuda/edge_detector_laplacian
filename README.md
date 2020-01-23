# objects detector infrastructure based on classic method (sharpening + laplacian + thresholding)
the program load an image and manipulate the output.
this program not classify the object, just edge markering it.

# libraries
- cv2
- numpy as np
- skimage.exposure

# example
there is a car image from google search just for the test, you cn change the image.

# parameters
- marker_width is the width of the points markering the edges
- threshold is the intensity of the pixel after applying convolution with the laplacian kernel

# FYI
you can change the sharpening kernel and the laplacian kernel to whatever you want.
suggestion:
- sobel
- canny (output the edges)
