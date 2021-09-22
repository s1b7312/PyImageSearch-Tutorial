
# import packages
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args['image'])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# edge detection
edged = cv2.Canny(gray, 30, 150, apertureSize=3)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# thresholding
# set pixels less than 225 to 255 and invert the colors
thresh = cv2.threshold(gray, thresh=225, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# detecting and drawing contours (i.e., outlines of the foreground objects)
cnts = cv2.findContours(image=thresh.copy(), mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE)[0]
# cnts = imutils.grab_contours(cnts)
output = image.copy()

# loop over the contours
# OR cv2.drawContours(image=output, contours=cnts, contourIdx=-1,
#                      color=(240, 0, 159), thickness=3)
for c in cnts:
    cv2.drawContours(image=output, contours=[c], contourIdx=-1,
                     color=(240, 0, 159), thickness=3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

# draw the total number of contours found
text = f"I found {len(cnts)} objects!"
cv2.putText(img=output, text=text, org=(10, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
            color=(240, 0, 159), thickness=2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# erosions and dilations are typically used to reduce noise in binary images (a side-effect of thresholding)
# this is typically useful in removing small blobs in the mask image

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(src=mask, kernel=None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the objects
mask = thresh.copy()
mask = cv2.dilate(src=mask, kernel=None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# masking and bitwise operations
# use the threshold image and mask the original image
# apply bitwise AND

mask = thresh.copy()
output = cv2.bitwise_and(src1=image, src2=image, mask=mask)
cv2.imshow("Masked", output)
cv2.waitKey(0)

