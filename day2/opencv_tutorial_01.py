
# import packages
import cv2
import imutils

# load image
image = cv2.imread("../images/jp.jpg")
(h, w, d) = image.shape
print(f"width={w}, height={h}, depth={d}")

# display the image
cv2.imshow("Image", image)
cv2.waitKey(0)

# get RGB values for the pixel at (50, 50)
# opencv images are in BGR order
(B, G, R) = image[50, 50]
print(f"R={R}, G={G}, B={B}")

# roi = image[20:120, 250:350]
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)

# resize to 200x200 with no regard for aspect ratio
# resized = cv2.resize(image, (200, 200))
# cv2.imshow("Fixed Resizing", resized)
# cv2.waitKey(0)

# maintain aspect ratio
r = 300.0 / w
dim = (300, int(h*r))
# resized = cv2.resize(image, dim)
resized = imutils.resize(image, width=300)
cv2.imshow("Apect Ratio Resized", resized)
cv2.waitKey(0)

# rotate clockwise by 45 degrees
centre = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centre, angle=-45, scale=1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# using imutils
# rotated = imutils.rotate(image, -45)

# rotate without clipping
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# smoothing an image
# apply Gaussian blur with 11x11 kernel
blurred = cv2.GaussianBlur(image, ksize=(11, 11), sigmaX=0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# drawing 2px bounding box
output = image.copy()
cv2.rectangle(output, pt1=(250, 20), pt2=(350, 120),
              color=(0, 0, 255), thickness=2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# draw a solid blue circle
output = image.copy()
cv2.circle(output, center=(300, 150), radius=20, color=(255, 0, 0), thickness=-1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw text on the image
output = image.copy()
cv2.putText(output, text="OpenCV + Jurassic Park!!!", org=(10, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0),
            thickness=2)
cv2.imshow("Text", output)
cv2.waitKey(0)
