import cv2
import numpy as np

img = cv2.imread('1original.jpg',0)

src = np.array([[50,50],[450,450],[70,420],[420,70]], np.float32)
dst = np.array([[0,0],[299,299],[0,299],[299,0]], np.float32)

ret = cv2.getPerspectiveTransform(src,dst)
print(ret)