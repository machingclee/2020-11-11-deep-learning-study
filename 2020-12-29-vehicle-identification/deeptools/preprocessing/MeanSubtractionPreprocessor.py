import cv2


class MeanSubtractionPreprocessor:
    def __init__(self, mean_R, mean_G, mean_B):
        self.mean_R = mean_R
        self.mean_G = mean_G
        self.mean_B = mean_B

    def preprocess(self, image):
        # if image is an integer array, then cv2 will perform modulo arithematics to make them nonnegative,
        # so we transform it into float array before splitting.
        (B, G, R) = cv2.split(image.astype("float32"))
        R = R - self.mean_R
        G = G - self.mean_G
        B = B - self.mean_B

        return cv2.merge([B, G, R])
