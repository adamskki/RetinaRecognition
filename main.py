import cv2
from skimage.data import camera
import numpy as np
from skimage.filters import frangi, hessian

SCALE=25

def calculate_precision(image, labeled, mask):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    all_pixels = 0

    for i in range(len(image)):
        for j in range(len(image[i])):
            if mask[i][j] == 0:
                image[i][j] = 0

    for i in range(len(image)):
        for j in range(len(image[i])):
            if labeled[i][j] == 255 and image[i][j] == 255 and mask[i][j] == 255:
                true_positive += 1
            elif labeled[i][j] == 255 and image[i][j] == 0 and mask[i][j] == 255:
                false_negative += 1
            elif labeled[i][j] == 0 and image[i][j] == 255 and mask[i][j] == 255:
                false_positive += 1
            elif labeled[i][j] == 0 and image[i][j] == 0 and mask[i][j] == 255:
                true_negative += 1

    all_pixels = true_negative + true_positive + false_positive + false_negative

    print("all: ", all_pixels)
    print("true positive: ", true_positive)
    print("false positive: ", false_positive)
    print("true negative: ", true_negative)
    print("false negative: ", false_negative)
    print("accuracy: ", (true_positive + true_negative) / all_pixels)
    print("sensitivity: ", true_positive / (true_positive + false_negative))
    print("specificity: ", true_negative / (false_positive + true_negative))

def main():
    img = cv2.imread('/images/1.jpg', cv2.IMREAD_COLOR)

    manual = cv2.imread('/manual/1.tif', cv2.IMREAD_COLOR)

    mask = cv2.imread('/mask/1.tif', cv2.IMREAD_COLOR)

    width = int(img.shape[1] * SCALE / 100)
    height = int(img.shape[0] * SCALE / 100)
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_manual = cv2.resize(manual, dim, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)



    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(resized_img, -1, kernel_sharpening)

    temp =  cv2.fastNlMeansDenoisingColored(sharpened,None,3,10,7,21)

    grey_scale = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(grey_scale, None, 6, 7, 21)
    th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8)

    # kernel = np.ones((1, 1), np.uint8)
    # erosion = cv2.erode(th3, kernel, iterations=1)
    # dilation = cv2.dilate(erosion, kernel, iterations=1)

    kernel = np.ones((1, 1), np.uint8)
    end = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=7)

    grey_manual = cv2.cvtColor(resized_manual, cv2.COLOR_BGR2GRAY)
    grey_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

    calculate_precision(end,grey_manual,grey_mask)
    cv2.imshow('Image Sharpening', end)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()