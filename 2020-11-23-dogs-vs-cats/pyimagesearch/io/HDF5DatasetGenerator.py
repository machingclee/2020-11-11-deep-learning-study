from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarized=True, numOfClasses=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarized = binarized
        self.numOfClasses = numOfClasses

        self.db = h5py.File(dbPath, "r")
        self.numOfImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        # this function generate a list of batches of data a
        # passes = number of epoch that we want
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.numOfImages, self.batchSize):
                images = self.db["images"][i:i + self.batchSize]
                labels = self.db["labels"][i:i + self.batchSize]

                if self.binarized:
                    labels = to_categorical(labels, self.numOfClasses)

                if self.preprocessors is not None:
                    procImages = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        procImages.append(image)

                    images = np.array(procImages)

                if self.aug is not None:
                    (images, labels) = next(
                        self.aug.flow(images, labels, batch_size=self.batchSize))

                # therefore .generator() gives an array of batched images and batched labels in run time:
                yield (images, labels)

            epochs = epochs + 1

    def close(self):
        self.db.close()
