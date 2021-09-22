
# import necessary packages
import time
import argparse
import numpy as np

import cv2
import imutils
from imutils.video import VideoStream

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pretrained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="threshold to filter weak detections")
args = vars(ap.parse_args())

# load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# frame count
f = 0

# loop over the frames from the video stream
while True:
    # print(f"Frame: {f}")
    f += 1

    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob and get detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args['confidence']:
            continue

        # compute bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw bounding box on the image
        text = "{:.2f}%".format(confidence * 100)
        y = startY-10 if startY-10 > 10 else startY+10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # if 'q' is pressed, break from the loop
    if key == ord("q"):
        break

    # cleanup
    cv2.destroyAllWindows()
    vs.stop()
