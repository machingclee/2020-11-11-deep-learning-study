import cv2
import numpy as np
import os


class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = [] if preprocessors == None else preprocessors

    def load(self, imagePaths, verbose=-1):
        images = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors != None:
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

            images.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

        return (np.array(images), np.array(labels))
