# usage: python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch-80.hdf5
# usage: python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch-80.hdf5 --vdeo mp4/baby.mp4

from sys import flags
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video", help="path to the optional video file")

args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])


while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((220, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 0:
        rect = sorted(rects, reverse=True, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))[0]
        (f_x, f_y, f_w, f_h) = rect
        roi = gray_img[f_y:f_y+f_h, f_x:f_x+f_w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        probs = model.predict(roi)[0]
        label = EMOTIONS[probs.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, probs)):
            text = "{}: {:2f}%".format(emotion, prob)
            w = int(prob * 300)

            cv2.rectangle(
                canvas,
                (5, i*35 + 5),
                (w, i*35 + 35),
                (0, 0, 255),
                -1
            )
            cv2.putText(
                canvas, text,
                (10, i*35 + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frameClone,
                label,
                (f_x, f_y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2
            )
            cv2.rectangle(
                frameClone,
                (f_x, f_y),
                (f_x+f_w, f_y+f_h),
                (0, 0, 255),
                2
            )

        cv2.imshow("Face", frameClone)
        cv2.imshow("Probabilities", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()
