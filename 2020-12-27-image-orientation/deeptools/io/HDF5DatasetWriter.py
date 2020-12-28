import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufferSize=1000):
        """
        dims = shape of training dataset

        outputPath = destination of our hdf5 db file

        dataKey = name of the data to be stored in hdf5 format, like images, features, etc

        bufferSize = size of our in-memory buffer, default to save 1000 feature vectors/images
        """

        if os.path.exists(outputPath):
            raise ValueError(
                "The supplied outoutPath already exists and cannot be overwritten.")

        # "w": create file, truncate if exists
        self.db = h5py.File(outputPath, "w")
        # note that self.data, self.labels and later labelSet in self.storeClassLabels are all just reference
        # to self.db's properties. They are from time to time being rewritten (like >>) in self.flush method
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        self.bufferSize = bufferSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # array.extend(array), concatenate two arrays
        # both append and extend modify the original array, no return
        # i.e., a = [1], then a.extend([2]), we have a = [1, 2]
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufferSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # vlen=unicode for python2
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),),
                                          dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()
