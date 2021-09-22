
# import packages
import cv2
import numpy as np

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros(shape=(4, 2), dtype='float32')

    # the top-left point will have the smallest sum
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    # now, compute the differences between the coordinates
    # the top-right point will have the smallest difference
    # the bottom left point will have the largest difference
    # diff = pts.diff(axis=1)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of points and unpack them
    rect = order_points(pts)
    (tl, tr, bl, br) = rect

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right x-coordinates,
    # or the top-right and top-left x-coordinates
    widthA = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    widthB = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    maxWidth = max(int(widthA), int(widthB))

    # compute the height similarly
    heightA = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
    heightB = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
    maxHeight = max(int(heightA), int(heightB))

    # construct matrix to contain the birds-eye view of the image
    # specifying points in the order:
    # top-left, top-right, bottom-left, bottom-right
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]
    ], dtype=np.float32)

    # compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(src=image, M=M, dsize=(maxWidth, maxHeight))

    return warped