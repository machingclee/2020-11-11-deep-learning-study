import imutils
import cv2

import os

# cv2.imread is an np.ndarray
# imutils.resize take np.ndarray to np.ndarray
# if only height or width is provided, imutils.resize resize proportionally
# cv2.resize on the other hand requires both width and height as its second argument (turple).


class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # assume w < h, our final height will be, with h_0 = image.shape[0],
        # h_0 - dH * 2 = h_0 - [(h_0 - self.height)/2] * 2 =  self.height
        # the case that h <= w is similar.
        if w < h:
            image = imutils.resize(image,
                                   width=self.width,
                                   inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image,
                                   height=self.height,
                                   inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # update newly resized h and w.
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image,
                          (self.width, self.height),
                          interpolation=self.inter)
