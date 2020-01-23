import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def convolve(image, kernel, threshold):
    # get the dimensions of the image and the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]
    indices = []
    # "pad" the borders of the input image so the spatial size won't reduce due to size of the kernel
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the region of interest of the image by extracting the kernel pixel from the image
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the convolution between the ROI and the kernel and sum to K
            k = (roi * kernel).sum()
            if k > threshold:
                indices.append((x, y, x-pad, y-pad))
            # store the convolved value in the output (x,y)- coordinate of the output image
            output[y - pad, x - pad] = k
    # rescale the output image to be in the range [0, 1] from [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255)
    return output, indices


if __name__ == '__main__':
    # construct the Laplacian kernel used to detect edge-like
    # regions of an image
    sharpening = np.array((
        [-2, -2, -2],
        [-2, 29, -2],
        [-2, -2, -2]), dtype="int")
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

    img = cv2.imread('car.png')
    convolve_sharpening, indices = convolve(img, sharpening, 255)
    # apply the laplacian kernel to the image using convolution with threshold of 80
    convolve_laplacian, indices = convolve(convolve_sharpening, laplacian, 200)

    cv2.imshow("original ", img)

    for x1, y1, x2, y2 in indices:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("original with edge detection", img)
    cv2.imshow("laplacian - convolution", convolve_laplacian)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
