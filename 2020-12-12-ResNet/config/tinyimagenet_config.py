import os


def join_path(*arg):
    return os.path.sep.join(arg)


N_CLASSES = 200


PROJECT_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])


TEST_HDF5_PATH = join_path(PROJECT_DIR, "hdf5", "test.hdf5")
TRAIN_HDF5_PATH = join_path(PROJECT_DIR, "hdf5", "train.hdf5")
VAL_HDF5_PATH = join_path(PROJECT_DIR, "hdf5", "val.hdf5")

MEAN_JSON_PATH = join_path(PROJECT_DIR, "output", "tiny-image-net-200-mean.json")
