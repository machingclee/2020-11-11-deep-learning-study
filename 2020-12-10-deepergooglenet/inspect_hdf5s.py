from config import tiny_imagenet_config as config
import h5py
import os

fileNames = ["train.hdf5", "val.hdf5", "test.hdf5"]
filePaths = [os.path.sep.join([config.PROJECT_DIR, "hdf5", fileName]) for fileName in fileNames]

for filePath in filePaths:
    db = h5py.File(filePath, "r")
    print(db["images"].shape)
    db.close()
