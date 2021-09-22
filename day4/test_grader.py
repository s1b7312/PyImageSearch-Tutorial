
import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform

# define the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# define the correct answer keys
answer_key = {0:1, 1:4, 2:0, 3:3, 4:1}

# load the image, convert it to grayscale, blur it slightly,
# and find the edges
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
