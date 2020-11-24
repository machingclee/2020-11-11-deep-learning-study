import numpy as np
import cv2


class CropPreprocessor:
    def __init__(self, width, height, flipped=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.flipped = flipped
        self.inter = inter

    def preprocess(self, image):
        crops = []
        (h, w) = image.shape[:2]
        squares = [[0, 0, self.width, self.height],
                   [w - self.width, 0, w, self.height]
                   [w - self.width, h - self.height, w, h]
                   [0, h-self.height, self.width, h]
                   ]

        middle_startX = int(0.5 * (w - self.width))
        middle_startY = int(0.5 * (h - self.height))
        squares.append([middle_startX, middle_startY,
                        middle_startX + self.width, middle_startY + self.height])

        for (startX, startY, endX, endY) in squares:
            crop = image[startY:endY, startX:endX]
            # this is just to make sure the is no pixel difference, even 1 pixel:
            crop = cv2.resize(crop,
                              (self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        if self.flipped:
            # from doc, sec parameter indicated flipping direction
            # 1: horizontal, 0: vertical, -1: both
            mirros = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirros)

        return np.array(crops)
