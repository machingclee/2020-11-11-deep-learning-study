from tensorflow.keras.preprocessing.image import img_to_array

# this class mainly deals with PIL Image instance.


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)
