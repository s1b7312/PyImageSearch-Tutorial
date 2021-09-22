
# import packages
import cv2
import imutils
import argparse

import numpy as np

from skimage.filters import threshold_local

from transform import four_point_transform

# define the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height to the new height,
# clone it, and resize it
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)   # make height=500 and maintain aspect ratio

# convert the image to grayscale, blur it and find the edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
edged = cv2.Canny(gray, threshold1=75, threshold2=200)

# display the image and the detected edges
print("STEP 1: Edge detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(curve=c, closed=True)
    approx = cv2.approxPolyDP(curve=c, epsilon=0.02*peri, closed=True)

    # if the approximated contour has four points, we can assume that we have
    # found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# show the contour
print("STEP 2: Find contours of the paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four-point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give a 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, block_size=11, method='gaussian')
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
